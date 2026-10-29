//! JSON-lines interface for local parity checks; not a public network service.
use hanzo_active_inference::Model;
use serde::Deserialize;
use std::io::{self, BufRead};
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Request {
    a: Vec<Vec<Vec<f64>>>,
    b: Vec<Vec<Vec<f64>>>,
    c: Vec<f64>,
    belief: Vec<f64>,
    policies: Vec<Vec<usize>>,
    habits: Vec<f64>,
    allowed: Vec<bool>,
    precision: f64,
}
fn main() {
    for line in io::stdin().lock().lines() {
        let result = (|| -> Result<_, Box<dyn std::error::Error>> {
            let line = line?;
            if line.len() > 4 * 1024 * 1024 {
                return Err("request too large".into());
            }
            let r: Request = serde_json::from_str(&line)?;
            Ok(Model::new(r.a, r.b, r.c)?.policies(
                &r.belief,
                &r.policies,
                &r.habits,
                &r.allowed,
                r.precision,
            )?)
        })();
        match result {
            Ok(value) => println!("{}", serde_json::to_string(&value).unwrap()),
            Err(e) => println!("{}", serde_json::json!({"error":e.to_string()})),
        }
    }
}
