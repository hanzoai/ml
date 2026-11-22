use hanzo_ml::{DType, Device};
use hanzo_nn::{VarBuilder, VarMap};
use hanzo_train::raft::{
    evaluate_zen_raft, prepare_raft_samples, train_zen_raft, DomainQaRecord, Passage, RaftConfig,
    ZenRaftModel,
};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::sync::Arc;
use std::thread;

fn build_domain_qa_corpus(
    num_records: usize,
    num_answers: usize,
    seed: u64,
) -> Vec<DomainQaRecord> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut out = Vec::with_capacity(num_records);

    for i in 0..num_records {
        let entity = i % num_answers;
        let query_mode_dynamic = i % 2 == 0;
        let answer = if query_mode_dynamic {
            rng.random_range(0..num_answers)
        } else {
            entity
        };

        let ent_tok = (4 + entity) as u32;
        let mode_tok = if query_mode_dynamic { 2u32 } else { 3u32 };
        let ans_tok = (24 + answer) as u32;

        let query_tokens = vec![mode_tok, ent_tok, ent_tok, mode_tok, 1, 1];
        let oracle_passage = Passage {
            doc_id: format!("orc-{i}"),
            tenant_id: "tenant-zen".to_string(),
            tokens: vec![mode_tok, ent_tok, ans_tok, ans_tok, ent_tok, ans_tok],
            is_oracle: true,
        };

        let mut distractors = Vec::with_capacity(4);
        for d in 0..4 {
            let wrong_ent = (entity + 1 + d) % num_answers;
            let wrong_ans = (answer + 1 + d) % num_answers;
            let w_ent_tok = (4 + wrong_ent) as u32;
            let w_ans_tok = (24 + wrong_ans) as u32;
            distractors.push(Passage {
                doc_id: format!("dist-{i}-{d}"),
                tenant_id: "tenant-zen".to_string(),
                tokens: vec![
                    rng.random_range(2..=3),
                    w_ent_tok,
                    w_ans_tok,
                    w_ans_tok,
                    w_ent_tok,
                    w_ans_tok,
                ],
                is_oracle: false,
            });
        }

        out.push(DomainQaRecord {
            tenant_id: "tenant-zen".to_string(),
            query_tokens,
            oracle_passage,
            distractor_passages: distractors,
            answer_label: answer as u32,
        });
    }
    out
}

#[test]
fn raft_outperforms_no_rag_sft_and_naive_oracle_rag_on_matched_eval() {
    let base_cfg = RaftConfig {
        base_model_license: "Apache-2.0".to_string(),
        vocab_size: 64,
        num_answers: 8,
        hidden_dim: 48,
        num_passages: 4,
        seq_len: 6,
        oracle_keep_prob: 0.75,
        citation_loss_weight: 1.0,
        learning_rate: 0.015,
        steps: 240,
        batch_size: 32,
    };

    let train_records = build_domain_qa_corpus(256, base_cfg.num_answers, 42);
    let eval_records = build_domain_qa_corpus(160, base_cfg.num_answers, 9001);
    let eval_mixed = prepare_raft_samples(&eval_records, &base_cfg, 0.50, 777).unwrap();

    // 1. Baseline A: Parametric Domain SFT without RAG context (use_context = false)
    let sft_train = prepare_raft_samples(&train_records, &base_cfg, 0.0, 101).unwrap();
    let (sft_vm, sft_model, sft_ms, _) =
        train_zen_raft(&sft_train, &base_cfg, false, None, 201).unwrap();
    let sft_rep = evaluate_zen_raft(
        &sft_model,
        &sft_vm,
        &eval_mixed,
        &base_cfg,
        false,
        sft_ms,
        0,
    )
    .unwrap();

    // 2. Baseline B: Naive RAG fine-tuning (P = 1.0, oracle ALWAYS present during training)
    let mut naive_cfg = base_cfg.clone();
    naive_cfg.oracle_keep_prob = 1.0;
    let naive_train = prepare_raft_samples(&train_records, &naive_cfg, 1.0, 102).unwrap();
    let (naive_vm, naive_model, naive_ms, _) =
        train_zen_raft(&naive_train, &naive_cfg, true, None, 202).unwrap();
    let naive_rep = evaluate_zen_raft(
        &naive_model,
        &naive_vm,
        &eval_mixed,
        &naive_cfg,
        true,
        naive_ms,
        0,
    )
    .unwrap();

    // 3. RAFT (P = 0.75: 75% oracle + distractors, 25% distractor-only robustness supervision)
    let tmp_dir = std::env::temp_dir().join("hanzo_zen_raft_test");
    let ckpt_path = tmp_dir.join("zen_raft_p075.safetensors");
    let raft_train =
        prepare_raft_samples(&train_records, &base_cfg, base_cfg.oracle_keep_prob, 103).unwrap();
    let (raft_vm, raft_model, raft_ms, ckpt_bytes) =
        train_zen_raft(&raft_train, &base_cfg, true, Some(&ckpt_path), 203).unwrap();
    let raft_rep = evaluate_zen_raft(
        &raft_model,
        &raft_vm,
        &eval_mixed,
        &base_cfg,
        true,
        raft_ms,
        ckpt_bytes,
    )
    .unwrap();

    // 4. Negative control: corrupted/shuffled labels falsify genuine retrieval learning
    let mut neg_samples = raft_train.clone();
    for (idx, s) in neg_samples.iter_mut().enumerate() {
        s.answer_label = ((idx * 5 + 1) % base_cfg.num_answers) as u32;
        s.citation_target = (idx % (base_cfg.num_passages + 1)) as u32;
    }
    let (neg_vm, neg_model, neg_ms, _) =
        train_zen_raft(&neg_samples, &neg_cfg(&base_cfg), true, None, 204).unwrap();
    let neg_rep =
        evaluate_zen_raft(&neg_model, &neg_vm, &eval_mixed, &base_cfg, true, neg_ms, 0).unwrap();

    println!(
        "RAFT_BENCHMARK no_rag_sft: overall_qa={:.3} oracle_qa={:.3} dist_qa={:.3} cite={:.3} reject={:.3} false_cite={:.3} ms={:.1}",
        sft_rep.overall_qa_accuracy,
        sft_rep.oracle_present_qa_accuracy,
        sft_rep.distractor_only_qa_accuracy,
        sft_rep.citation_support_accuracy,
        sft_rep.distractor_rejection_rate,
        sft_rep.false_citation_rate,
        sft_rep.train_time_ms
    );
    println!(
        "RAFT_BENCHMARK naive_rag_p100: overall_qa={:.3} oracle_qa={:.3} dist_qa={:.3} cite={:.3} reject={:.3} false_cite={:.3} ms={:.1}",
        naive_rep.overall_qa_accuracy,
        naive_rep.oracle_present_qa_accuracy,
        naive_rep.distractor_only_qa_accuracy,
        naive_rep.citation_support_accuracy,
        naive_rep.distractor_rejection_rate,
        naive_rep.false_citation_rate,
        naive_rep.train_time_ms
    );
    println!(
        "RAFT_BENCHMARK zen_raft_p075: overall_qa={:.3} oracle_qa={:.3} dist_qa={:.3} cite={:.3} reject={:.3} false_cite={:.3} params={} ckpt_bytes={} ms={:.1} ms_step={:.3}",
        raft_rep.overall_qa_accuracy,
        raft_rep.oracle_present_qa_accuracy,
        raft_rep.distractor_only_qa_accuracy,
        raft_rep.citation_support_accuracy,
        raft_rep.distractor_rejection_rate,
        raft_rep.false_citation_rate,
        raft_rep.param_count,
        raft_rep.checkpoint_bytes,
        raft_rep.train_time_ms,
        raft_rep.ms_per_step
    );
    println!(
        "RAFT_BENCHMARK neg_control: overall_qa={:.3} cite={:.3}",
        neg_rep.overall_qa_accuracy, neg_rep.citation_support_accuracy
    );

    assert!(
        raft_rep.oracle_present_qa_accuracy >= 0.85,
        "RAFT must exploit oracle passages effectively: got {:.3}",
        raft_rep.oracle_present_qa_accuracy
    );
    assert!(
        raft_rep.citation_support_accuracy >= 0.85,
        "RAFT must cite the shuffled oracle passage accurately: got {:.3}",
        raft_rep.citation_support_accuracy
    );
    assert!(
        raft_rep.distractor_rejection_rate > naive_rep.distractor_rejection_rate + 0.40,
        "RAFT (P=0.75) must reject distractors far better than Naive RAG (P=1.0): raft={:.3} naive={:.3}",
        raft_rep.distractor_rejection_rate,
        naive_rep.distractor_rejection_rate
    );
    assert!(
        raft_rep.overall_qa_accuracy > sft_rep.overall_qa_accuracy + 0.12,
        "RAFT must beat No-RAG SFT on mixed evaluation: raft={:.3} sft={:.3}",
        raft_rep.overall_qa_accuracy,
        sft_rep.overall_qa_accuracy
    );
    assert!(
        raft_rep.overall_qa_accuracy > naive_rep.overall_qa_accuracy + 0.12,
        "RAFT must beat Naive P=1.0 RAG on mixed evaluation: raft={:.3} naive={:.3}",
        raft_rep.overall_qa_accuracy,
        naive_rep.overall_qa_accuracy
    );
    assert!(
        raft_rep.citation_support_accuracy > sft_rep.citation_support_accuracy + 0.80,
        "No-RAG SFT cannot cite retrieved passages"
    );
    assert!(
        raft_rep.overall_qa_accuracy > neg_rep.overall_qa_accuracy + 0.40,
        "RAFT must beat shuffled-label negative control"
    );
    assert!(ckpt_bytes > 1024, "safetensors checkpoint must be written");

    // 5. Verify bit-identical reload from safetensors and concurrent multi-thread evaluation
    let mut reloaded_vm = VarMap::new();
    let dev = Device::Cpu;
    let reloaded_vb = VarBuilder::from_varmap(&reloaded_vm, DType::F32, &dev);
    let reloaded_model = Arc::new(ZenRaftModel::new(reloaded_vb, &base_cfg).unwrap());
    reloaded_vm.load(&ckpt_path).unwrap();
    let reloaded_vm = Arc::new(reloaded_vm);
    let shared_eval = Arc::new(eval_mixed);
    let shared_cfg = Arc::new(base_cfg);

    let mut handles = Vec::new();
    for _ in 0..4 {
        let m = Arc::clone(&reloaded_model);
        let vm = Arc::clone(&reloaded_vm);
        let ev = Arc::clone(&shared_eval);
        let c = Arc::clone(&shared_cfg);
        handles.push(thread::spawn(move || {
            evaluate_zen_raft(&m, &vm, &ev, &c, true, 0.0, ckpt_bytes).unwrap()
        }));
    }
    for h in handles {
        let r = h.join().unwrap();
        assert!((r.overall_qa_accuracy - raft_rep.overall_qa_accuracy).abs() < 1e-9);
        assert!((r.citation_support_accuracy - raft_rep.citation_support_accuracy).abs() < 1e-9);
        assert!((r.distractor_rejection_rate - raft_rep.distractor_rejection_rate).abs() < 1e-9);
    }

    let _ = std::fs::remove_dir_all(&tmp_dir);
}

fn neg_cfg(cfg: &RaftConfig) -> RaftConfig {
    cfg.clone()
}

#[test]
fn prepare_batch_builds_padded_tensors_and_masks_without_stubs() {
    let cfg = RaftConfig {
        base_model_license: "Apache-2.0".into(),
        vocab_size: 32,
        num_answers: 4,
        hidden_dim: 16,
        num_passages: 3,
        seq_len: 5,
        oracle_keep_prob: 0.75,
        citation_loss_weight: 1.0,
        learning_rate: 1e-2,
        steps: 4,
        batch_size: 2,
    };
    let records = vec![
        hanzo_train::raft::DomainQaRecord {
            tenant_id: "tenant-alpha".into(),
            query_tokens: vec![2, 5, 7],
            oracle_passage: hanzo_train::raft::Passage {
                doc_id: "orc-1".into(),
                tenant_id: "tenant-alpha".into(),
                tokens: vec![11, 12, 13, 14],
                is_oracle: true,
            },
            distractor_passages: vec![
                hanzo_train::raft::Passage {
                    doc_id: "d-1".into(),
                    tenant_id: "tenant-alpha".into(),
                    tokens: vec![3, 4],
                    is_oracle: false,
                },
                hanzo_train::raft::Passage {
                    doc_id: "d-2".into(),
                    tenant_id: "tenant-alpha".into(),
                    tokens: vec![6, 8, 9],
                    is_oracle: false,
                },
                hanzo_train::raft::Passage {
                    doc_id: "d-3".into(),
                    tenant_id: "tenant-alpha".into(),
                    tokens: vec![15],
                    is_oracle: false,
                },
            ],
            answer_label: 2,
        },
        hanzo_train::raft::DomainQaRecord {
            tenant_id: "tenant-alpha".into(),
            query_tokens: vec![4, 9, 10, 11, 12],
            oracle_passage: hanzo_train::raft::Passage {
                doc_id: "orc-2".into(),
                tenant_id: "tenant-alpha".into(),
                tokens: vec![20, 21],
                is_oracle: true,
            },
            distractor_passages: vec![
                hanzo_train::raft::Passage {
                    doc_id: "d-4".into(),
                    tenant_id: "tenant-alpha".into(),
                    tokens: vec![1, 2, 3],
                    is_oracle: false,
                },
                hanzo_train::raft::Passage {
                    doc_id: "d-5".into(),
                    tenant_id: "tenant-alpha".into(),
                    tokens: vec![7, 8],
                    is_oracle: false,
                },
                hanzo_train::raft::Passage {
                    doc_id: "d-6".into(),
                    tenant_id: "tenant-alpha".into(),
                    tokens: vec![18, 19, 22, 23, 24],
                    is_oracle: false,
                },
            ],
            answer_label: 1,
        },
    ];
    let samples = hanzo_train::raft::prepare_raft_samples(&records, &cfg, 0.5, 42).unwrap();
    let batch = hanzo_train::raft::prepare_batch(&samples, &cfg, &Device::Cpu).unwrap();
    assert_eq!(batch.query_ids.dims(), &[2, 5]);
    assert_eq!(batch.passage_ids.dims(), &[2, 3, 5]);
    assert_eq!(batch.attention_mask.dims(), &[2, 20]);
    assert_eq!(batch.prompt_loss_mask.dims(), &[2, 22]);
    assert_eq!(batch.answer_targets.dims(), &[2]);
    assert_eq!(batch.citation_targets.dims(), &[2]);

    let mask_rows = batch.prompt_loss_mask.to_vec2::<f32>().unwrap();
    for row in mask_rows {
        assert!(row[..20].iter().all(|&v| v == 0.0));
        assert_eq!(row[20], 1.0);
        assert_eq!(row[21], 1.0);
    }
}
