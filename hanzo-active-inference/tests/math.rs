use hanzo_active_inference::{Counts, Model};
fn identity() -> Vec<Vec<f64>> {
    vec![vec![1., 0.], vec![0., 1.]]
}
fn model() -> Model {
    Model::new(
        vec![identity(), vec![vec![0.5, 0.5], vec![0.5, 0.5]]],
        vec![identity(), identity()],
        vec![0.5, 0.5],
    )
    .unwrap()
}
#[test]
fn epistemic_action_is_preferred() {
    let m = model();
    let p = m
        .policies(
            &[0.5, 0.5],
            &[vec![0], vec![1]],
            &[0.5, 0.5],
            &[true, true],
            4.,
        )
        .unwrap();
    assert_eq!(p.selected, 0);
    assert!(p.probabilities[0] > 0.9);
    assert!((p.terms[0].information_gain - 2f64.ln()).abs() < 1e-12);
    assert!((p.terms[1].information_gain).abs() < 1e-12);
}
#[test]
fn decompositions_match() {
    for q in [vec![0., 1.], vec![0.2, 0.8], vec![0.5, 0.5]] {
        for action in 0..2 {
            let t = model().terms(&q, action).unwrap();
            assert!((t.risk + t.ambiguity - t.extrinsic + t.information_gain).abs() < 1e-12);
        }
    }
}
#[test]
fn bayesian_posterior_and_free_energy() {
    let p = model().observe(&[0.2, 0.8], 0, 0).unwrap();
    assert_eq!(p.belief, vec![1., 0.]);
    assert!((p.free_energy + 0.2f64.ln()).abs() < 1e-12);
    assert!(model().observe(&[0., 1.], 0, 0).is_err());
}
#[test]
fn hard_mask_beats_information_value() {
    let p = model()
        .policies(
            &[0.5, 0.5],
            &[vec![0], vec![1]],
            &[0.5, 0.5],
            &[false, true],
            100.,
        )
        .unwrap();
    assert_eq!(p.probabilities, vec![0., 1.]);
    assert!(model()
        .policies(&[0.5, 0.5], &[vec![0]], &[1.], &[false], 1.)
        .is_err());
}
#[test]
fn reject_nonfinite_ragged_and_unnormalized() {
    assert!(Model::new(vec![identity()], vec![identity()], vec![0., 1.]).is_err());
    assert!(Model::new(vec![identity()], vec![vec![vec![1.]]], vec![0.5, 0.5]).is_err());
    assert!(model().predict(&[0.2, 0.7], 0).is_err());
    assert!(model().predict(&[f64::NAN, 0.], 0).is_err());
    assert!(model()
        .policies(&[0.5, 0.5], &[vec![0]], &[1.], &[true], f64::INFINITY)
        .is_err());
}
#[test]
fn count_learning_is_finite_conservative_and_immutable() {
    let counts = Counts {
        a: vec![vec![vec![1., 1.], vec![1., 1.]]],
        b: vec![vec![vec![1., 1.], vec![1., 1.]]],
    };
    let learned = counts
        .learned(&[0.2, 0.8], 0, 0, 1., vec![0.5, 0.5])
        .unwrap();
    assert_eq!(counts.a[0][0], vec![1., 1.]);
    assert!((learned.a[0].iter().flatten().sum::<f64>() - 5.).abs() < 1e-12);
    assert!((learned.b[0].iter().flatten().sum::<f64>() - 5.).abs() < 1e-12);
    assert!((learned.b[0][0][0] - 1.1).abs() < 1e-12);
    assert!(learned.model(vec![0.5, 0.5]).is_ok());
}
#[test]
fn horizon_and_prior_change_policy_posterior() {
    let p = model()
        .policies(
            &[0.5, 0.5],
            &[vec![0, 0], vec![0]],
            &[0.1, 0.9],
            &[true, true],
            0.,
        )
        .unwrap();
    assert!((p.probabilities[0] - 0.1).abs() < 1e-12);
}
#[test]
fn variational_bound_is_tight_only_at_posterior() {
    let id = identity();
    let m = Model::new(
        vec![vec![vec![0.9, 0.2], vec![0.1, 0.8]]],
        vec![id],
        vec![0.5, 0.5],
    )
    .unwrap();
    let posterior = m.observe(&[0.4, 0.6], 0, 0).unwrap();
    let optimum = m
        .variational_free_energy(&posterior.belief, &[0.4, 0.6], 0, 0)
        .unwrap();
    assert!((optimum - posterior.free_energy).abs() < 1e-12);
    assert!(
        m.variational_free_energy(&[0.5, 0.5], &[0.4, 0.6], 0, 0)
            .unwrap()
            > optimum
    );
}
