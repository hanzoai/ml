//! `soup` — merge safetensors checkpoints, or compress/restore a fine-tune as a
//! 1-bit BitDelta, using the native `hanzo_ml::model_delta` implementation.
//!
//! Examples:
//!
//! ```text
//! # Uniform model soup of three checkpoints.
//! soup average -o merged.safetensors a.safetensors b.safetensors c.safetensors
//!
//! # Weighted soup (weights are normalised internally).
//! soup average -o merged.safetensors -w 1,3,1 a.safetensors b.safetensors c.safetensors
//!
//! # Delta soup: base + mean(ft_i - base).
//! soup delta -b base.safetensors -o merged.safetensors ft1.safetensors ft2.safetensors
//!
//! # BitDelta: compress a fine-tune to ~1 bit/weight, then restore it.
//! soup bitdelta-encode -b base.safetensors -f ft.safetensors -o ft.bitdelta
//! soup bitdelta-apply  -b base.safetensors -d ft.bitdelta -o restored.safetensors
//! ```

use clap::{Parser, Subcommand};
use hanzo_ml::model_delta;
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(
    name = "soup",
    about = "Model soups + BitDelta over safetensors (native hanzo-ml)",
    version
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Average matching tensors across several checkpoints (model soup).
    Average {
        /// Input checkpoints to average (>= 1).
        #[arg(required = true)]
        models: Vec<PathBuf>,
        /// Output safetensors path.
        #[arg(short, long)]
        out: PathBuf,
        /// Optional comma-separated per-model weights, e.g. `1,3,1`.
        /// Normalised internally; omit for a uniform average.
        #[arg(short, long, value_delimiter = ',')]
        weights: Option<Vec<f32>>,
    },
    /// Average the deltas (ft_i - base) and add the mean back onto base (delta soup).
    Delta {
        /// Base checkpoint.
        #[arg(short, long)]
        base: PathBuf,
        /// Fine-tuned checkpoints (>= 1).
        #[arg(required = true)]
        finetunes: Vec<PathBuf>,
        /// Output safetensors path.
        #[arg(short, long)]
        out: PathBuf,
    },
    /// Compress a fine-tune relative to base into a 1-bit `.bitdelta` file.
    BitdeltaEncode {
        /// Base checkpoint.
        #[arg(short, long)]
        base: PathBuf,
        /// Fine-tuned checkpoint to compress.
        #[arg(short, long)]
        finetuned: PathBuf,
        /// Output `.bitdelta` path.
        #[arg(short, long)]
        out: PathBuf,
    },
    /// Reconstruct a fine-tune from base + a `.bitdelta` file.
    BitdeltaApply {
        /// Base checkpoint.
        #[arg(short, long)]
        base: PathBuf,
        /// Input `.bitdelta` file.
        #[arg(short, long)]
        delta: PathBuf,
        /// Output safetensors path.
        #[arg(short, long)]
        out: PathBuf,
    },
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::Average {
            models,
            out,
            weights,
        } => {
            model_delta::soup(&models, weights.as_deref(), &out)?;
            println!(
                "soup: averaged {} checkpoint(s) -> {}",
                models.len(),
                out.display()
            );
        }
        Command::Delta {
            base,
            finetunes,
            out,
        } => {
            model_delta::delta_soup(&base, &finetunes, &out)?;
            println!(
                "soup: base {} + mean delta of {} fine-tune(s) -> {}",
                base.display(),
                finetunes.len(),
                out.display()
            );
        }
        Command::BitdeltaEncode {
            base,
            finetuned,
            out,
        } => {
            model_delta::encode(&base, &finetuned, &out)?;
            let bd = model_delta::BitDelta::from_file(&out)?;
            println!(
                "bitdelta: encoded {} tensor(s) from {} vs {} -> {}",
                bd.header.tensors.len(),
                finetuned.display(),
                base.display(),
                out.display()
            );
        }
        Command::BitdeltaApply { base, delta, out } => {
            model_delta::decode_apply(&base, &delta, &out)?;
            println!(
                "bitdelta: applied {} to {} -> {}",
                delta.display(),
                base.display(),
                out.display()
            );
        }
    }
    Ok(())
}
