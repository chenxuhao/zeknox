// Copyright 2024 OKX Group
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

// Batched forward NTT benchmark over Goldilocks.
//
// Usage:
//   cargo run --release --features=gl64 --example ntt_bench -- \
//       [--batches 16] [--log-min 10] [--log-max 24] [--iters 20] [--warmup 3] [--gpu 0] [--no-verify]
//
// For each log size it reports two timings:
//   device: inputs and outputs stay in GPU memory (kernel time + launch overhead)
//   host:   inputs and outputs in host memory (adds HtoD + DtoH transfers and device alloc)
// Before timing, the GPU output of the first and last polynomial in the batch is
// checked against Plonky2's CPU FFT (skipped for log sizes > 20 or with --no-verify).

use plonky2_field::{
    fft::fft,
    goldilocks_field::GoldilocksField,
    polynomial::PolynomialCoeffs,
    types::{Field, PrimeField64},
};
use rand::random;
use std::time::Instant;
use zeknox::{
    device::memory::HostOrDeviceSlice, init_twiddle_factors_rs, ntt_batch, types::NTTConfig,
};

const GL_MODULUS: u64 = 0xffffffff00000001;
const MAX_VERIFY_LOG: usize = 20;

struct Args {
    batches: usize,
    log_min: usize,
    log_max: usize,
    iters: usize,
    warmup: usize,
    gpu: usize,
    verify: bool,
}

fn parse_args() -> Args {
    let mut args = Args {
        batches: 16,
        log_min: 10,
        log_max: 24,
        iters: 20,
        warmup: 3,
        gpu: 0,
        verify: true,
    };
    let mut it = std::env::args().skip(1);
    while let Some(flag) = it.next() {
        if flag == "--no-verify" {
            args.verify = false;
            continue;
        }
        let val: usize = it
            .next()
            .and_then(|v| v.parse().ok())
            .unwrap_or_else(|| panic!("missing or invalid value for {}", flag));
        match flag.as_str() {
            "--batches" => args.batches = val,
            "--log-min" => args.log_min = val,
            "--log-max" => args.log_max = val,
            "--iters" => args.iters = val,
            "--warmup" => args.warmup = val,
            "--gpu" => args.gpu = val,
            _ => panic!("unknown flag {}", flag),
        }
    }
    assert!(args.log_min >= 1 && args.log_min <= args.log_max && args.log_max <= 32);
    assert!(args.batches >= 1 && args.iters >= 1);
    args
}

fn random_fr() -> u64 {
    random::<u64>() % GL_MODULUS
}

fn cpu_fft(input: &[u64]) -> Vec<u64> {
    let coeffs = input
        .iter()
        .map(|x| GoldilocksField::from_canonical_u64(*x))
        .collect::<Vec<GoldilocksField>>();
    fft(PolynomialCoeffs { coeffs })
        .values
        .iter()
        .map(|x| x.to_canonical_u64())
        .collect()
}

fn verify(gpu: usize, log_n: usize, batches: usize, input: &[u64]) {
    let n = 1usize << log_n;
    let mut buf = input.to_vec();
    let mut cfg = NTTConfig::default();
    cfg.batches = batches as u32;
    ntt_batch(gpu, buf.as_mut_ptr(), log_n, cfg);

    for b in [0, batches - 1] {
        let expected = cpu_fft(&input[b * n..(b + 1) * n]);
        assert_eq!(
            &buf[b * n..(b + 1) * n],
            &expected[..],
            "GPU NTT mismatch vs Plonky2 at log_n={} batch={}",
            log_n,
            b
        );
    }
}

/// Runs `f` warmup + iters times; returns (median, min) in milliseconds.
fn time_ms<F: FnMut()>(warmup: usize, iters: usize, mut f: F) -> (f64, f64) {
    for _ in 0..warmup {
        f();
    }
    let mut samples: Vec<f64> = (0..iters)
        .map(|_| {
            let t = Instant::now();
            f();
            t.elapsed().as_secs_f64() * 1e3
        })
        .collect();
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    (samples[samples.len() / 2], samples[0])
}

fn main() {
    let args = parse_args();
    println!(
        "Goldilocks batched NTT: batches={}, log_n={}..={}, iters={}, warmup={}, gpu={}",
        args.batches, args.log_min, args.log_max, args.iters, args.warmup, args.gpu
    );
    println!(
        "{:>5} {:>12} {:>14} {:>12} {:>14} {:>12}  {}",
        "log_n", "elements", "device_med_ms", "device_min", "host_med_ms", "host_min", "verified"
    );

    for log_n in args.log_min..=args.log_max {
        let n = 1usize << log_n;
        let total = n * args.batches;

        init_twiddle_factors_rs(args.gpu, log_n);

        let mut host_input: Vec<u64> = (0..total).map(|_| random_fr()).collect();

        let verified = if args.verify && log_n <= MAX_VERIFY_LOG {
            verify(args.gpu, log_n, args.batches, &host_input);
            "yes"
        } else {
            "skipped"
        };

        // Device-resident: the NTT runs in place, so repeated runs keep transforming
        // the same buffer; that does not affect timing.
        let mut device_data: HostOrDeviceSlice<'_, u64> =
            HostOrDeviceSlice::cuda_malloc(args.gpu as i32, total).unwrap();
        device_data.copy_from_host(&host_input).unwrap();
        let mut dev_cfg = NTTConfig::default();
        dev_cfg.batches = args.batches as u32;
        dev_cfg.are_inputs_on_device = true;
        dev_cfg.are_outputs_on_device = true;
        let (dev_med, dev_min) = time_ms(args.warmup, args.iters, || {
            ntt_batch(args.gpu, device_data.as_mut_ptr(), log_n, dev_cfg.clone())
        });
        drop(device_data);

        // Host-resident: includes host<->device transfers.
        let mut host_cfg = NTTConfig::default();
        host_cfg.batches = args.batches as u32;
        let (host_med, host_min) = time_ms(args.warmup, args.iters, || {
            ntt_batch(args.gpu, host_input.as_mut_ptr(), log_n, host_cfg.clone())
        });

        println!(
            "{:>5} {:>12} {:>14.3} {:>12.3} {:>14.3} {:>12.3}  {}",
            log_n, total, dev_med, dev_min, host_med, host_min, verified
        );
    }
}
