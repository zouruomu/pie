// Teacher-forced logits: the chat-rendered prompt as text-completion-bench renders it, then `teacher`
// fed one decode step at a time; each position's logits go to the client as base64 f32 messages.
use inferlet::eta::prelude::*;
use inferlet::{chat, session};
use serde::{Deserialize, Serialize};

#[derive(Deserialize)]
struct Input {
    prompt: String,
    #[serde(default = "default_system")]
    system: String,
    teacher: Vec<u32>,
}

fn default_system() -> String {
    "You are a helpful benchmarking assistant.".into()
}

#[derive(Serialize)]
struct Output {
    prompt_tokens: usize,
    positions: usize,
}

fn b64(bytes: &[u8]) -> String {
    const T: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut s = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for c in bytes.chunks(3) {
        let n = (u32::from(c[0]) << 16) | (u32::from(*c.get(1).unwrap_or(&0)) << 8) | u32::from(*c.get(2).unwrap_or(&0));
        for i in 0..4 {
            if i <= c.len() {
                s.push(T[(n >> (18 - 6 * i) & 63) as usize] as char);
            } else {
                s.push('=');
            }
        }
    }
    s
}

// In pieces of 32768 values: a message the client's socket takes whole.
fn send_logits(v: &[f32]) {
    for piece in v.chunks(32768) {
        let bytes: Vec<u8> = piece.iter().flat_map(|x| x.to_le_bytes()).collect();
        session::send(&b64(&bytes));
    }
}

#[inferlet::main]
async fn main(input: Input) -> Result<Output> {
    let vocab = model::output_vocab_size();
    let mut prompt = chat::system_user(&input.system, &input.prompt);
    prompt.extend(chat::cue_thinking(true));
    let n = prompt.len() as u32;
    let ws = WorkingSet::new();
    let pipe = Pipeline::new();

    let toks: Vec<i32> = prompt.iter().map(|&t| t as i32).collect();
    let toks_p = Channel::from(&toks[..]);
    let indptr_p = Channel::from([0u32, n]);
    let lg_p = Channel::new([vocab], dtype::f32);
    let fwd_p = ForwardPass::new();
    fwd_p.embed(&toks_p, &indptr_p)?;
    fwd_p.state(&[&ws], StateGeometry::folding())?;
    let out_p = lg_p.clone();
    fwd_p.epilogue(move || {
        out_p.put(&reshape(intrinsics::output(0, "logits"), [vocab]));
    });
    fwd_p.submit(&pipe)?;
    let first: Vec<f32> = lg_p.take_host().await?;
    send_logits(&first);

    for &t in &input.teacher {
        let tok = Channel::from([t as i32]);
        let indptr = Channel::from([0u32, 1u32]);
        let lg = Channel::new([vocab], dtype::f32);
        let fwd = ForwardPass::new();
        fwd.embed(&tok, &indptr)?;
        fwd.state(&[&ws], StateGeometry::folding())?;
        let out = lg.clone();
        fwd.epilogue(move || {
            out.put(&reshape(intrinsics::output(0, "logits"), [vocab]));
        });
        fwd.submit(&pipe)?;
        let v: Vec<f32> = lg.take_host().await?;
        send_logits(&v);
    }
    Ok(Output { prompt_tokens: prompt.len(), positions: input.teacher.len() + 1 })
}
