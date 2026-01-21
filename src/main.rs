use std::borrow::BorrowMut;
use std::cmp::Ordering;
use std::fs::OpenOptions;
use std::io::Write;
use std::num::NonZeroUsize;
use std::path::PathBuf;

use anyhow::anyhow;
use clap::{Parser, ValueEnum};
use ndarray::Axis;
use ndarray_stats::QuantileExt;
use ort::{CUDAExecutionProvider, ExecutionProvider, Session, Tensor};
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use rayon::slice::ParallelSlice;
use rust_tokenizers::tokenizer::{BertTokenizer, Tokenizer, TruncationStrategy};
use rust_tokenizers::vocab::{BertVocab, Vocab};
use rust_tokenizers::Mask;

type ShapeAndVector = ([usize; 2], Vec<i64>);

fn pad_and_flatten_vectors(values: Vec<Vec<i64>>) -> anyhow::Result<ShapeAndVector> {
    let nrows = values.len();
    let ncols = values.iter().map(|t| t.len()).max().unwrap_or(0);
    let data = values
        .into_iter()
        .map(|t| zero_pad_vec_to_length(t, ncols))
        .collect::<Result<Vec<Vec<_>>, _>>()?;
    let data = data.into_iter().flatten().collect();
    anyhow::Ok(([nrows, ncols], data))
}

const ZERO_512: [i64; 512] = [0_i64; 512];

fn zero_pad_vec_to_length(vec: Vec<i64>, ncols: usize) -> Result<Vec<i64>, anyhow::Error> {
    let l = vec.len();
    match l.partial_cmp(&ncols) {
        Some(Ordering::Less) => Ok([&vec, &ZERO_512[..(ncols - l)]].concat()),
        Some(Ordering::Equal) => Ok(vec),
        Some(Ordering::Greater) => Err(anyhow!(format!("Vector to long to pad: {l} > {ncols}!"))),
        None => panic!("Invalid partial_cmp while padding vector!"),
    }
}

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, ValueEnum, Debug)]
enum Aggregation {
    None,
    Last,
    Strict,
}

#[allow(clippy::upper_case_acronyms)]
#[derive(Debug, PartialEq, Eq)]
enum Entity {
    MISC,
    PER,
    ORG,
    LOC,
}

#[derive(Debug)]
enum Label {
    O,
    B(Entity),
    I(Entity),
}

impl From<usize> for Label {
    fn from(value: usize) -> Self {
        match value {
            0 => Label::O,
            1 => Label::B(Entity::MISC),
            2 => Label::I(Entity::MISC),
            3 => Label::B(Entity::PER),
            4 => Label::I(Entity::PER),
            5 => Label::B(Entity::ORG),
            6 => Label::I(Entity::ORG),
            7 => Label::B(Entity::LOC),
            8 => Label::I(Entity::LOC),
            _ => panic!("Label index out of bounds: {value}"),
        }
    }
}

impl serde::Serialize for Label {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        match self {
            Label::O => serializer.serialize_str("O"),
            // Label::B(e) => serializer.serialize_str(&format!("B-{e:?}")),
            // Label::I(e) => serializer.serialize_str(&format!("I-{e:?}")),
            Label::B(e) => serializer.serialize_str(&format!("{e:?}")),
            Label::I(e) => serializer.serialize_str(&format!("{e:?}")),
        }
    }
}

#[derive(serde::Serialize, Debug)]
struct Annotation {
    label: Label,
    begin: u32,
    end: u32,
}

impl Annotation {
    fn new(label: Label, begin: u32, end: u32) -> Self {
        Annotation { label, begin, end }
    }
}

#[allow(clippy::upper_case_acronyms)]
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, ValueEnum, Debug)]
enum ImplementedProviders {
    CPU,
    CUDA,
}

const DEFAULT_MODEL_PATH: &str = "data/model.onnx";
const DEFAULT_VOCAB_PATH: &str = "data/vocab.txt";

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(short, long, default_value = "cpu")]
    device: ImplementedProviders,

    #[arg(long, default_value_t = 0)]
    device_id: usize,

    #[arg(short, long)]
    threads: Option<usize>,

    #[arg(short, long, default_value_t = 128)]
    batch_size: usize,

    #[arg(short, long, default_value = DEFAULT_MODEL_PATH)]
    model: PathBuf,

    #[arg(short, long, default_value = DEFAULT_VOCAB_PATH)]
    vocab: PathBuf,

    #[arg(short, long, value_enum, default_value = "last")]
    aggregation: Aggregation,

    #[arg()]
    corpus: PathBuf,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    let builder = Session::builder()?;

    let threads = args
        .threads
        .map_or_else(
            || std::thread::available_parallelism().map_err(anyhow::Error::from),
            |v| NonZeroUsize::new(v).ok_or(anyhow!("Number of threads must be > 0!")),
        )?
        .get();

    match args.device {
        ImplementedProviders::CPU => (),
        ImplementedProviders::CUDA => {
            let cuda = CUDAExecutionProvider::default().with_device_id(args.device_id as i32);
            if let Err(err) = cuda.register(&builder) {
                Err(anyhow!(
                    "Failed to register CUDA execution provider: {err:?}"
                ))?
            }
        }
    }
    let corpus_path = args.corpus;
    let model_path = args.model;
    let vocab_path = args.vocab;

    ort::init()
        // .with_execution_providers([CPUExecutionProvider::default().build()])
        .commit()?;

    let corpus: Vec<String> = std::fs::read_to_string(corpus_path)
        .unwrap()
        .lines()
        .map(String::from)
        .filter(|s| !s.is_empty())
        .collect();

    let vocab = BertVocab::from_file(vocab_path)?;
    let tokenizer = BertTokenizer::from_existing_vocab(vocab, false, false);

    let session = builder
        // .with_optimization_level(GraphOptimizationLevel::Level1)?
        .with_parallel_execution(threads > 1)?
        .with_intra_threads(threads)?
        .commit_from_file(model_path)?;

    let annotations = run_prediction(
        corpus,
        args.batch_size,
        args.aggregation,
        tokenizer,
        session,
    );
    let json_string = serde_json::to_string_pretty(&annotations?).unwrap();
    OpenOptions::new()
        .write(true)
        .open("ort-rs.json")?
        .write_all(json_string.as_bytes())?;
    Ok(())
}

fn run_prediction(
    corpus: Vec<String>,
    chunk_size: usize,
    aggregation: Aggregation,
    tokenizer: BertTokenizer,
    session: Session,
) -> anyhow::Result<Vec<Vec<Annotation>>> {
    let annotations: Vec<Vec<Annotation>> = corpus
        .par_chunks(chunk_size)
        .map(|batch| tokenizer.encode_list(batch, 512, &TruncationStrategy::LongestFirst, 0))
        .map(|tokens| {
            let (input_ids, attention_mask) = get_input_ids_and_attention_masks(&tokens);

            (input_ids, attention_mask, tokens)
        })
        .collect::<Vec<_>>()
        .into_iter()
        .map(
            |((input_shape, input_ids), (attention_shape, attention_mask), tokens)| {
                let input_ids = Tensor::from_array((input_shape, input_ids))?;
                let attention_mask = Tensor::from_array((attention_shape, attention_mask))?;
                let outputs = session.run(ort::inputs![input_ids, attention_mask]?)?;
                let preds: Vec<Vec<usize>> = outputs[0]
                    .try_extract_tensor::<f32>()?
                    .view()
                    .map_axis(Axis(2), |v| v.argmax())
                    .rows()
                    .into_iter()
                    .map(|r| r.to_vec().into_iter().collect::<Result<Vec<usize>, _>>())
                    .collect::<Result<Vec<Vec<usize>>, _>>()?;
                Ok((tokens, preds))
            },
        )
        .collect::<Result<Vec<_>, anyhow::Error>>()?
        .into_par_iter()
        .flat_map(|(tokens, preds)| {
            (tokens, preds)
                .into_par_iter()
                .map(|(ts, ps)| process_outputs(ts, ps, &aggregation))
                .collect::<Vec<Vec<Annotation>>>()
        })
        .collect();
    Ok(annotations)
}

fn get_input_ids_and_attention_masks(
    batch_encoding: &[rust_tokenizers::TokenizedInput],
) -> (ShapeAndVector, ShapeAndVector) {
    let mut input_ids: Vec<Vec<i64>> = Vec::with_capacity(batch_encoding.len());
    let mut attention_mask: Vec<Vec<i64>> = Vec::with_capacity(batch_encoding.len());
    for seq_encoding in batch_encoding.iter() {
        input_ids.push(seq_encoding.token_ids.clone());
        attention_mask.push(vec![1_i64; seq_encoding.token_ids.len()]);
    }
    let input_ids = pad_and_flatten_vectors(input_ids).unwrap();
    let attention_mask = pad_and_flatten_vectors(attention_mask).unwrap();
    (input_ids, attention_mask)
}

fn process_outputs(
    tokens: rust_tokenizers::TokenizedInput,
    predictions: Vec<usize>,
    aggregation: &Aggregation,
) -> Vec<Annotation> {
    let mut annotations: Vec<Annotation> = Vec::new();
    let mut last_annotation: Option<Annotation> = None;

    let iterable = std::iter::zip(
        std::iter::zip(&tokens.token_offsets, &tokens.mask),
        &predictions,
    );
    for ((offset, mask), pred) in iterable {
        if let Some(offset) = offset {
            match mask {
                Mask::Special => continue,
                Mask::Continuation => {
                    if let Some(la) = last_annotation.borrow_mut() {
                        la.end = offset.end;
                    } else {
                        panic!(
                            "Got a continuation token without preceding ordinary token! {tokens:#?} {predictions:#?}"
                        )
                    }
                }
                _ => {
                    let label = Label::from(*pred);
                    if let Some(mut la) = last_annotation {
                        match (aggregation, &label, &la.label) {
                            (Aggregation::Last, Label::I(_), Label::B(_) | Label::I(_)) => {
                                la.borrow_mut().end = offset.end;
                                last_annotation = Some(la);
                                continue;
                            }
                            (Aggregation::Strict, Label::I(e), Label::B(n) | Label::I(n))
                                if e == n =>
                            {
                                la.borrow_mut().end = offset.end;
                                last_annotation = Some(la);
                                continue;
                            }
                            // TODO: Merge arms below?
                            (Aggregation::None, _, _) => annotations.push(la),
                            // TODO: Emit a warning if an Inside tag follows an Outside tag?
                            (_, Label::I(_), Label::O) => annotations.push(la),
                            _ => annotations.push(la),
                        }
                    }
                    last_annotation = Some(Annotation::new(label, offset.begin, offset.end));
                }
            };
        }
    }

    annotations
        .into_iter()
        .filter(|a| !matches!(a.label, Label::O))
        .collect()
}
