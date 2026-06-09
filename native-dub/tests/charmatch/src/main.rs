//! charmatch <ref> <hyp>  ->  prints a char-match fraction in [0,1].
//!
//! Both args may be a literal string or `@path` to read from a file. The metric is designed
//! for the native-dub ASR/round-trip checks where the hypothesis legitimately CONTAINS the
//! reference (the demo clip's audio is longer than the known reference sentence) and where the
//! two transcripts differ only in punctuation and the occasional homophone:
//!
//!   1. strip punctuation + whitespace from both (CJK and ASCII punctuation),
//!   2. take the first `len(ref)` characters of the hypothesis (prefix alignment),
//!   3. report the fraction of positions whose characters agree.
//!
//! If the hypothesis is SHORTER than the reference, the denominator stays `len(ref)` so a
//! truncated transcription is penalized. Comparison is over Unicode scalar values (chars), so
//! it is correct for Chinese as well as English. Pure Rust on purpose: the e2e suite greps the
//! run log for `python`, and we don't want the test harness itself to trip that check.

use std::fs;

fn read_arg(a: &str) -> String {
    if let Some(path) = a.strip_prefix('@') {
        fs::read_to_string(path).unwrap_or_default()
    } else {
        a.to_string()
    }
}

fn is_punct_or_space(c: char) -> bool {
    if c.is_whitespace() {
        return true;
    }
    // ASCII punctuation
    if c.is_ascii_punctuation() {
        return true;
    }
    // CJK / fullwidth punctuation commonly emitted by ASR (commas, periods, quotes, etc.)
    matches!(c,
        '\u{3000}'        // ideographic space
        | '\u{3001}'      // 、
        | '\u{3002}'      // 。
        | '\u{FF0C}'      // ，
        | '\u{FF01}'      // ！
        | '\u{FF1F}'      // ？
        | '\u{FF1A}'      // ：
        | '\u{FF1B}'      // ；
        | '\u{2018}' | '\u{2019}' | '\u{201C}' | '\u{201D}'   // smart quotes
        | '\u{2026}'      // ...
        | '\u{2014}' | '\u{2013}'                              // dashes
        | '\u{300C}' | '\u{300D}' | '\u{300E}' | '\u{300F}'   // 「」『』
        | '\u{FF08}' | '\u{FF09}'                              // fullwidth parens
    )
}

fn normalize(s: &str) -> Vec<char> {
    s.chars().filter(|&c| !is_punct_or_space(c)).collect::<String>()
        .to_lowercase()
        .chars()
        .collect()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: charmatch <ref|@file> <hyp|@file>");
        std::process::exit(2);
    }
    let refs = normalize(&read_arg(&args[1]));
    let hyp = normalize(&read_arg(&args[2]));

    if refs.is_empty() {
        // empty reference is a configuration error, not a 100% match
        println!("0.0");
        return;
    }

    let n = refs.len();
    let mut agree = 0usize;
    for i in 0..n {
        if i < hyp.len() && hyp[i] == refs[i] {
            agree += 1;
        }
    }
    let frac = agree as f64 / n as f64;
    println!("{frac:.4}");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_match_after_punct_strip() {
        // identical content, hyp adds CJK punctuation -> 1.0
        let r = "每个人到了一定年纪";
        let h = "每个人，到了。一定年纪";
        let rv = normalize(r);
        let hv = normalize(h);
        let agree = (0..rv.len()).filter(|&i| i < hv.len() && hv[i] == rv[i]).count();
        assert_eq!(agree, rv.len());
    }

    #[test]
    fn ref_is_prefix_of_longer_hyp() {
        // the real case: ASR output is longer than the known first-sentence reference,
        // differing only by one homophone (地 vs 的). Expect >= 0.95.
        let r = "每个人到了一定年纪一切都看淡了顺其自然地活着珍惜所有的遇见";
        let h = "每个人到了一定年纪，一切都看淡了，顺其自然的活着，珍惜所有的遇见。笑对离开你的人";
        let rv = normalize(r);
        let hv = normalize(h);
        let agree = (0..rv.len()).filter(|&i| i < hv.len() && hv[i] == rv[i]).count();
        let frac = agree as f64 / rv.len() as f64;
        assert!(frac >= 0.95, "frac={frac}");
        assert!(frac < 1.0, "homophone should not be a perfect match");
    }

    #[test]
    fn truncated_hyp_is_penalized() {
        let r = "abcdefghij";
        let h = "abcde";
        let rv = normalize(r);
        let hv = normalize(h);
        let agree = (0..rv.len()).filter(|&i| i < hv.len() && hv[i] == rv[i]).count();
        let frac = agree as f64 / rv.len() as f64;
        assert!((frac - 0.5).abs() < 1e-9, "frac={frac}");
    }

    #[test]
    fn english_case_insensitive() {
        let rv = normalize("Hello World");
        let hv = normalize("hello, world!");
        assert_eq!(rv, hv);
    }
}
