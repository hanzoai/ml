//! `hanzo-train` CLI: train a DSpark draft on the DeepSpec target cache on the cluster runtime
//! ([`hanzo_train::cluster`]) and save an engine-loadable checkpoint. `fit` coordinates the run
//! and trains in it; `join` adds a machine with its own copy of the cache and init checkpoint.
//! The goal is a real, decreasing loss curve on real data + a checkpoint whose keys and shapes
//! exactly match the engine loader.

use std::path::PathBuf;

use anyhow::Result;
use clap::{Parser, Subcommand};

use hanzo_train::cluster::{self, host, Model, Params};
use hanzo_train::dspark::{self, Engine, Local, Spec, CODE};
use hanzo_train::gpu;

#[derive(Parser)]
#[command(
    name = "hanzo-train",
    about = "Native-Rust DSpark draft trainer, on one machine or many"
)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

// parsed once, so the size of `Fit` costs nothing
#[allow(clippy::large_enum_variant)]
#[derive(Subcommand)]
enum Cmd {
    /// Coordinate a run on `--listen` and train in it; other machines `join` it.
    Fit {
        #[command(flatten)]
        local: Local,
        #[command(flatten)]
        spec: Spec,
        /// Output checkpoint directory (writes model.safetensors + config.json).
        #[arg(long, default_value = "./dspark-mvp-ckpt")]
        out: PathBuf,
        /// Where workers reach the coordinator, host:port.
        #[arg(long, default_value = "127.0.0.1:0")]
        listen: String,
        /// Free memory (percent) below which training exits rather than take the machine down.
        #[arg(long, default_value_t = 15)]
        floor: u32,
    },
    /// Train as a worker of the run a `fit` coordinates at `addr`, joining again whenever the
    /// link fails, until the run is done.
    Join {
        /// The coordinator, host:port.
        addr: String,
        #[command(flatten)]
        local: Local,
        /// Free memory (percent) below which training exits rather than take the machine down.
        #[arg(long, default_value_t = 15)]
        floor: u32,
    },
}

fn fit(local: Local, spec: Spec, out: PathBuf, listen: &str) -> Result<()> {
    let (run, begin, draft) = dspark::run(&spec, &local, Some(out))?;
    let cfg = draft.model.cfg();
    println!(
        "cache: {} samples, hidden={}, n_fused={}, layer_ids={:?}",
        draft.cache.len(),
        cfg.hidden,
        draft.cache.n_fused,
        cfg.target_layer_ids
    );
    println!(
        "model: layers={} heads={}/{} head_dim={} intermediate={} vocab={} block={} markov_rank={}",
        cfg.layers,
        cfg.heads,
        cfg.kv_heads,
        cfg.head_dim,
        cfg.intermediate,
        cfg.vocab,
        cfg.block,
        cfg.markov_rank
    );
    let params = Params::of(draft.vars(), |n| draft.rate(n).is_some());
    println!(
        "trainable: {} tensors, {} params{}; baseline CE = ln(vocab) = {:.4}",
        params.names.len(),
        params.len,
        if spec.freeze_markov {
            " [markov frozen]"
        } else {
            ""
        },
        (cfg.vocab as f64).ln()
    );
    let report = cluster::lead(listen, run, begin, Engine { spec, local }, draft)?;
    println!(
        "--- done: {} steps over {} rounds ---",
        report.merged, report.rounds
    );
    Ok(())
}

fn join(addr: &str, local: Local) -> Result<()> {
    let joined = cluster::join(cluster::resolve(addr)?, &host(), CODE, |setup| {
        dspark::load(setup, &local)
    })?;
    let s = match joined {
        Some(w) => w.run()?,
        None => Default::default(),
    };
    println!(
        "trained {} steps, {} tokens, over {} rounds and {} links",
        s.ids.len(),
        s.tokens,
        s.rounds,
        s.links
    );
    Ok(())
}

fn main() -> Result<()> {
    match Cli::parse().cmd {
        Cmd::Fit {
            local,
            spec,
            out,
            listen,
            floor,
        } => {
            gpu::guard(floor);
            fit(local, spec, out, &listen)
        }
        Cmd::Join { addr, local, floor } => {
            gpu::guard(floor);
            join(&addr, local)
        }
    }
}
