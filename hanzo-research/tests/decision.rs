//! Decision records on the wire: each body a POST sends carries exactly the fields the contract
//! names for it (camelCase, DDT's; the server's own fields left out), a DDT record inside its
//! envelope is sent as written, and each client method reaches its route under `/v1/` and reads
//! the answer the server sends. A commitment is only read: the client files none.

use std::collections::BTreeSet;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpListener;
use std::sync::{Arc, Mutex};
use std::thread;

use hanzo_research::{
    Action, Api, Charter, Commitment, Decision, Depend, Evaluation, Evidence, Link, Program,
    Recommendation, Source, Sysml,
};
use serde_json::{json, Value};

/// The keys of a JSON object, sorted.
fn keys(v: &Value) -> BTreeSet<&str> {
    v.as_object()
        .unwrap_or_else(|| panic!("{v} is not an object"))
        .keys()
        .map(String::as_str)
        .collect()
}

fn set<'a>(names: &[&'a str]) -> BTreeSet<&'a str> {
    names.iter().copied().collect()
}

const HASH: &str = "31513dddf264c7a242a5e1e32979a6e6d293cc9c579d3f21a4bd42e527a907ab";

fn evidence() -> Evidence {
    Evidence {
        uri: "techdata://sph/tsb/22".into(),
        hash: HASH.into(),
        kind: "document".into(),
        source: Source {
            system: "techdata".into(),
            revision: String::new(),
        },
        links: vec![Link {
            rel: "refines".into(),
            uri: "ddt:DEC-SPH-0412/need".into(),
        }],
        content: Some(json!({"bulletin": "TSB-22", "revision": "A"})),
        ..Evidence::default()
    }
}

/// A DDT-1.0 Decision record with its numbers as written: `3` stays an integer.
fn ddt() -> Value {
    json!({
        "id": "DEC-SPH-0412",
        "schemaVersion": "DDT-1.0",
        "program": "apu.select@1",
        "assumptions": [{"id": "BATTERY_MTTR_H", "value": 3, "range": [2, 8]}],
        "evidence": ["techdata://sph/tsb/22"]
    })
}

fn decision() -> Decision {
    Decision {
        decision: ddt(),
        charter: Charter {
            question: "Which auxiliary power source?".into(),
            scope: "vehicle concept".into(),
            owner: "Chief Engineer".into(),
            due: "2026-11-30".into(),
            criteria: vec!["silent watch".into()],
        },
        program: Some(Program {
            id: "apu.select".into(),
            version: 1,
            hash: format!("sha256:{HASH}"),
        }),
        ..Decision::default()
    }
}

fn evaluation() -> Evaluation {
    Evaluation {
        decision_id: "DEC-SPH-0412".into(),
        evaluation_run: json!({
            "id": "RUN_KAI_08B02A2DE43C89D7",
            "decisionId": "DEC-SPH-0412",
            "inputs": {"techdata://sph/tsb/22": format!("sha256:{HASH}")},
            "outputs": {"hash": "sha256:08b0"}
        }),
        package: json!({"hash": "sha256:08b0"}),
        recommendation: Recommendation {
            selection: Some("A2".into()),
            ranking: vec![json!({"option": "A2", "utility": 0.64})],
            flips: vec![],
            readiness: json!({"decisionClassPolicyId": "DCP", "verdict": "allow"}),
        },
        depends: vec![Depend {
            uri: "techdata://sph/tsb/22".into(),
            hash: HASH.into(),
        }],
        ..Evaluation::default()
    }
}

/// A DDT Action draft, as `program::ddt` exports it.
fn draft() -> Value {
    json!({
        "id": "RUN_KAI_08B02A2DE43C89D7/readiness",
        "decisionId": "DEC-SPH-0412",
        "status": "DRAFT",
        "verdict": "ask",
        "rule": "kai.missing",
        "description": "Is a test result or review that the charter requires missing for any option?"
    })
}

fn action() -> Action {
    Action {
        action: draft().as_object().unwrap().clone(),
        ..Action::default()
    }
}

#[test]
fn bodies_carry_the_contract_fields() {
    let e = serde_json::to_value(evidence()).unwrap();
    assert_eq!(
        keys(&e),
        set(&["uri", "hash", "kind", "source", "links", "content"])
    );
    assert_eq!(keys(&e["source"]), set(&["system"]));
    assert_eq!(keys(&e["links"][0]), set(&["rel", "uri"]));
    let revised = Source {
        system: "sysml".into(),
        revision: "c0ffee".into(),
    };
    assert_eq!(
        keys(&serde_json::to_value(revised).unwrap()),
        set(&["system", "revision"])
    );

    let s = serde_json::to_value(Sysml {
        project: "ltv".into(),
        commit: "c0ffee".into(),
        elements: vec![json!({"@id": "a", "@type": "Package"})],
    })
    .unwrap();
    assert_eq!(keys(&s), set(&["project", "commit", "elements"]));

    let d = serde_json::to_value(decision()).unwrap();
    assert_eq!(keys(&d), set(&["decision", "charter", "program"]));
    assert_eq!(
        keys(&d["charter"]),
        set(&["question", "scope", "owner", "due", "criteria"])
    );
    assert_eq!(keys(&d["program"]), set(&["id", "version", "hash"]));
    // The DDT record goes as written, its integers integers.
    assert_eq!(d["decision"], ddt());
    assert_eq!(
        serde_json::to_string(&d["decision"]["assumptions"][0]["value"]).unwrap(),
        "3"
    );
    let bare = Decision {
        charter: Charter {
            question: "Which?".into(),
            ..Charter::default()
        },
        program: None,
        ..decision()
    };
    let bare = serde_json::to_value(bare).unwrap();
    assert_eq!(keys(&bare), set(&["decision", "charter"]));
    assert_eq!(keys(&bare["charter"]), set(&["question"]));

    let v = serde_json::to_value(evaluation()).unwrap();
    assert_eq!(
        keys(&v),
        set(&[
            "decisionId",
            "evaluationRun",
            "package",
            "recommendation",
            "depends"
        ])
    );
    assert_eq!(
        keys(&v["recommendation"]),
        set(&["selection", "ranking", "flips", "readiness"])
    );
    assert_eq!(keys(&v["depends"][0]), set(&["uri", "hash"]));

    // An action is the DDT Action's fields, as written, and nothing of the server's.
    let a = serde_json::to_value(action()).unwrap();
    assert_eq!(a, draft());
    let assigned = Action {
        assignee: "z@hanzo.ai".into(),
        ..action()
    };
    let assigned = serde_json::to_value(assigned).unwrap();
    assert_eq!(
        keys(&assigned),
        set(&[
            "id",
            "decisionId",
            "status",
            "verdict",
            "rule",
            "description",
            "assignee"
        ])
    );
}

#[test]
fn records_read_back_with_the_servers_fields() {
    let held: Evidence = serde_json::from_value(json!({
        "id": "ev_2", "uri": "techdata://sph/tsb/22", "hash": HASH, "kind": "document",
        "source": {"system": "techdata"}, "links": null, "time": "2026-10-16T07:00:00Z",
        "version": 2, "by": "z@hanzo.ai"
    }))
    .unwrap();
    assert_eq!((held.version, held.links.len()), (2, 0));
    assert_eq!(held.content, None);

    let d: Decision = serde_json::from_value(json!({
        "id": "dec_2", "decision": ddt(), "charter": {"question": "Which?"},
        "version": 2, "supersedes": "dec_1",
        "depends": [{"uri": "techdata://sph/tsb/22"}, {"uri": "plm://sph/BOM/top@R2026.08"}],
        "time": "2026-10-16T07:00:00Z", "by": "z@hanzo.ai"
    }))
    .unwrap();
    assert_eq!((d.version, d.supersedes.as_str()), (2, "dec_1"));
    assert_eq!(d.depends[0].hash, "");
    assert_eq!(d.program, None);

    let a: Action = serde_json::from_value(json!({
        "id": "RUN_KAI_08B02A2DE43C89D7/readiness", "decisionId": "DEC-SPH-0412",
        "status": "AUTHORIZED", "verdict": "ask", "rule": "kai.missing", "description": "?",
        "assignee": "a@hanzo.ai", "authorizedBy": "z@hanzo.ai",
        "authorizedAt": "2026-10-16T07:00:00Z"
    }))
    .unwrap();
    assert_eq!(a.action["status"], "AUTHORIZED");
    assert_eq!(a.authorized_by, "z@hanzo.ai");
    assert!(!a.action.contains_key("authorizedBy"));

    let c: Commitment = serde_json::from_value(json!({
        "id": "com_1", "decisionId": "DEC-SPH-0412", "evaluationId": "evl_1",
        "selection": "A3", "note": "after the bulletin", "by": "z@hanzo.ai",
        "time": "2026-10-16T07:00:00Z"
    }))
    .unwrap();
    assert_eq!(
        (c.evaluation_id.as_str(), c.selection.as_str()),
        ("evl_1", "A3")
    );
}

// ── the stand-in ─────────────────────────────────────────────────────────────────────

/// Every request heard: method, target (path and query), body.
type Heard = Arc<Mutex<Vec<(String, String, Value)>>>;

/// A server answering each route as the contract says: a POST answers its record with the
/// server's fields filled; listings are `{data: [...]}`.
fn standin() -> (String, Heard) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let base = format!("http://{}", listener.local_addr().unwrap());
    let heard: Heard = Arc::default();
    let log = heard.clone();
    thread::spawn(move || {
        for stream in listener.incoming() {
            let mut s = stream.unwrap();
            let mut reader = BufReader::new(s.try_clone().unwrap());
            let mut line = String::new();
            reader.read_line(&mut line).unwrap();
            let mut parts = line.split_whitespace();
            let method = parts.next().unwrap_or_default().to_string();
            let target = parts.next().unwrap_or_default().to_string();
            let mut len = 0;
            loop {
                let mut h = String::new();
                reader.read_line(&mut h).unwrap();
                if h == "\r\n" || h.is_empty() {
                    break;
                }
                if let Some(v) = h.to_ascii_lowercase().strip_prefix("content-length:") {
                    len = v.trim().parse().unwrap_or(0);
                }
            }
            let mut body = vec![0; len];
            reader.read_exact(&mut body).unwrap();
            let body: Value = serde_json::from_slice(&body).unwrap_or(Value::Null);
            let answer = answer(&method, &target, &body);
            log.lock().unwrap().push((method, target, body));
            let bytes = answer.to_string();
            let _ = write!(
                s,
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{bytes}",
                bytes.len()
            );
        }
    });
    (base, heard)
}

/// `body` with the server's fields.
fn filed(body: &Value, id: &str) -> Value {
    let mut v = body.clone();
    v["id"] = json!(id);
    v["time"] = json!("2026-10-16T07:00:00Z");
    v["by"] = json!("z@hanzo.ai");
    v
}

fn answer(method: &str, target: &str, body: &Value) -> Value {
    let path = target.split('?').next().unwrap_or_default();
    match (method, path) {
        ("POST", "/v1/evidence") => {
            let mut v = filed(body, "ev_1");
            v["version"] = json!(1);
            v
        }
        ("POST", "/v1/evidence/sysml") => {
            let (p, c) = (&body["project"], &body["commit"]);
            let data: Vec<Value> = body["elements"]
                .as_array()
                .unwrap()
                .iter()
                .map(|e| {
                    json!({"id": "ev_s", "uri": format!("sysml://{}@{}#{}", p.as_str().unwrap(), c.as_str().unwrap(), e["@id"].as_str().unwrap()),
                           "hash": HASH, "kind": "element", "source": {"system": "sysml", "revision": c},
                           "links": [], "content": e, "version": 1})
                })
                .collect();
            json!({ "data": data })
        }
        ("GET", "/v1/evidence") => json!({"data": [
            {"id": "ev_2", "uri": "techdata://sph/tsb/22", "hash": "b2", "kind": "document",
             "source": {"system": "techdata"}, "links": [], "version": 2},
            {"id": "ev_1", "uri": "techdata://sph/tsb/22", "hash": HASH, "kind": "document",
             "source": {"system": "techdata"}, "links": [], "version": 1}
        ]}),
        ("GET", "/v1/evidence/dependents") => json!({
            "decisions": [{"id": "dec_1", "decisionId": "DEC-SPH-0412", "stale": false}],
            "evaluations": [{"id": "evl_1", "decisionId": "DEC-SPH-0412", "hash": HASH, "stale": true}]
        }),
        ("POST", "/v1/research/decisions") => {
            let mut v = filed(body, "dec_1");
            v["version"] = json!(1);
            v["depends"] = json!([{"uri": "techdata://sph/tsb/22"}]);
            v
        }
        ("POST", "/v1/research/evaluations") => filed(body, "evl_1"),
        ("POST", "/v1/research/actions") => body.clone(),
        _ => json!({"detail": "no route"}),
    }
}

#[test]
fn each_record_goes_to_its_route() {
    let (base, heard) = standin();
    let api = Api::new(&base, "kai", "hk-test".into()).unwrap();

    let e = api.evidence(&evidence()).unwrap();
    assert_eq!((e.id.as_str(), e.version), ("ev_1", 1));
    assert_eq!(e.content, evidence().content);

    let model = Sysml {
        project: "ltv".into(),
        commit: HASH.into(),
        elements: vec![json!({"@id": "e1", "@type": "PartUsage", "declaredName": "diesel"})],
    };
    let filed = api.sysml(&model).unwrap();
    assert_eq!(filed[0].uri, format!("sysml://ltv@{HASH}#e1"));

    let versions = api.versions("techdata://sph/tsb/22").unwrap();
    assert_eq!(
        versions.iter().map(|v| v.version).collect::<Vec<_>>(),
        [2, 1]
    );

    let deps = api.dependents("techdata://sph/tsb/22", Some(HASH)).unwrap();
    assert_eq!(deps.decisions[0].decision_id, "DEC-SPH-0412");
    assert!(!deps.decisions[0].stale && deps.evaluations[0].stale);

    let d = api.decision(&decision()).unwrap();
    assert_eq!((d.id.as_str(), d.version), ("dec_1", 1));
    assert_eq!(d.decision, ddt());
    assert_eq!(api.evaluation(&evaluation()).unwrap().id, "evl_1");
    assert_eq!(api.action(&action()).unwrap(), action());

    let heard = heard.lock().unwrap();
    let targets: Vec<(&str, &str)> = heard
        .iter()
        .map(|(m, t, _)| (m.as_str(), t.as_str()))
        .collect();
    assert_eq!(
        targets,
        [
            ("POST", "/v1/evidence"),
            ("POST", "/v1/evidence/sysml"),
            ("GET", "/v1/evidence?uri=techdata%3A%2F%2Fsph%2Ftsb%2F22"),
            (
                "GET",
                &*format!(
                    "/v1/evidence/dependents?uri=techdata%3A%2F%2Fsph%2Ftsb%2F22&hash={HASH}"
                )
            ),
            ("POST", "/v1/research/decisions"),
            ("POST", "/v1/research/evaluations"),
            ("POST", "/v1/research/actions"),
        ]
    );
    // Each POST sent the body the record serializes to.
    assert_eq!(heard[0].2, serde_json::to_value(evidence()).unwrap());
    assert_eq!(heard[1].2, serde_json::to_value(&model).unwrap());
    assert_eq!(heard[4].2, serde_json::to_value(decision()).unwrap());
    assert_eq!(heard[5].2, serde_json::to_value(evaluation()).unwrap());
    assert_eq!(heard[6].2, draft());
}
