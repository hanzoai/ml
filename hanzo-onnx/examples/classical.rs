//! Read a classical model — a random forest, a logistic regression, an XGBoost or
//! LightGBM export — and print what it answers.
//!
//! ```text
//! cargo run -p hanzo-onnx --example classical -- \
//!     hanzo-onnx/tests/ml/forest_clf.onnx 5.1,3.5,1.4,0.2 6.9,3.2,5.7,2.3
//! ```
//!
//! There are two verbs in this crate's whole surface for such a model: read the file,
//! and run it. Everything a classical model needs is in the file's attributes, so there
//! is no session to configure and no provider to choose.

use hanzo_ml::{Device, Result, Tensor};
use hanzo_onnx::{Key, Value};
use std::collections::HashMap;

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let Some(path) = args.next() else {
        eprintln!("usage: classical <model.onnx> <comma,separated,row> [row ...]");
        std::process::exit(2);
    };
    let rows: Vec<Vec<f32>> = args
        .map(|row| {
            row.split(',')
                .map(|v| v.trim().parse::<f32>())
                .collect::<std::result::Result<_, _>>()
        })
        .collect::<std::result::Result<_, _>>()
        .map_err(hanzo_ml::Error::wrap)?;
    if rows.is_empty() {
        eprintln!("give at least one row of features");
        std::process::exit(2);
    }
    let columns = rows[0].len();
    let flat: Vec<f32> = rows.iter().flatten().copied().collect();

    let model = hanzo_onnx::read_file(&path)?;
    let graph = model.graph.as_ref().expect("a model has a graph");
    let x = Tensor::from_vec(flat, (rows.len(), columns), &Device::Cpu)?;
    let outputs =
        hanzo_onnx::simple_eval(&model, HashMap::from([(graph.input[0].name.clone(), x)]))?;

    println!("{path}");
    // Report in the graph's own output order rather than a map's, so two runs read alike.
    for declared in &graph.output {
        let Some(value) = outputs.get(&declared.name) else {
            continue;
        };
        println!("  {}:", declared.name);
        match value {
            Value::Table(table) => {
                for r in 0..table.rows() {
                    let entries: Vec<String> = table
                        .row(r)?
                        .into_iter()
                        .map(|(key, score)| match key {
                            Key::Int(k) => format!("{k}: {score:.8}"),
                            Key::Text(k) => format!("{k}: {score:.8}"),
                        })
                        .collect();
                    println!("    row {r}  {{{}}}", entries.join(", "));
                }
            }
            Value::Text(text) => println!("    {:?}", text.elements()),
            Value::Tensor(t) => match t.dtype() {
                hanzo_ml::DType::I64 => println!("    {:?}", t.flatten_all()?.to_vec1::<i64>()?),
                _ => {
                    let t = t.to_dtype(hanzo_ml::DType::F32)?;
                    for (r, row) in t
                        .to_vec2::<f32>()
                        .unwrap_or_else(|_| {
                            vec![t
                                .flatten_all()
                                .and_then(|t| t.to_vec1())
                                .unwrap_or_default()]
                        })
                        .iter()
                        .enumerate()
                    {
                        println!("    row {r}  {row:?}");
                    }
                }
            },
        }
    }
    Ok(())
}
