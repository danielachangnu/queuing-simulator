use noisy_float::prelude::*;
use rand::prelude::*;
use rand_distr::{Beta, ChiSquared, Exp, InverseGaussian, Normal, Pareto};

use std::f64::INFINITY;
const EPSILON: f64 = 1e-8;
use std::f64::consts::PI;

struct Job {
    rem_size: f64,
    arrival_time: f64,
}

//Only insertion policies
#[derive(Debug)]
enum Policy {
    FCFS,
}
impl Policy {
    fn insertion_index(&mut self, queue: &Vec<Job>, new_job: &Job) -> usize {
        match self {
            Policy::FCFS => queue.len(),
        }
    }

    fn reorder(&mut self, queue: Vec<Job>) -> Vec<Job> {
        match self {
            Policy::FCFS => queue,
        }
    }
}

struct Results {
    response_times: Vec<usize>,
    step: f64,
    total_response_time: f64,
    total_queue_time: f64,
    num_jobs: usize,
    total_service_time: f64,
}

impl Results {
    pub fn mean_response_time(&self) -> f64 {
        self.total_response_time / (self.num_jobs as f64)
    }

    pub fn mean_queue_time(&self) -> f64 {
        self.total_queue_time / (self.num_jobs as f64)
    }

    pub fn mean_service_time(&self) -> f64 {
        self.total_service_time / (self.num_jobs as f64)
    }
}

fn simulate(
    lambda: f64,
    dist: Dist,
    policy: &mut Policy,
    step: f64,
    num_jobs: usize,
    seed: u64,
) -> Results {
    assert!((dist.mean() - 1.0).abs() < EPSILON); // mean = lambda is like the load, so if the mean is around 1 its accurate. if not, its not accurate
    let mut rng = StdRng::seed_from_u64(seed);
    let mut time = 0.0; // easy way to track response time 
    let arrival_dist = Exp::new(lambda).unwrap(); 
    let mut next_arrival = rng.sample(arrival_dist);
    let mut num_completions = 0;
    let mut queue: Vec<Job> = vec![];
let mut results = Results {
        step,
        response_times: vec![],
        total_response_time: 0.0,
        total_queue_time: 0.0,
        num_jobs,
        total_service_time: 0.0,
    };

    while num_completions < num_jobs {
        let next_event_diff =
            (next_arrival - time).min(queue.first().map_or(INFINITY, |j| j.rem_size));
        let was_arrival  = next_event_diff == (next_arrival - time);
        time += next_event_diff;
        if !queue.is_empty() {
            let job = queue.first_mut().unwrap();
            job.rem_size -= next_event_diff;
            if job.rem_size <= EPSILON {
                let job = queue.remove(0);
                let response = time - job.arrival_time;
                let service = job.original_size;
                let wait_time = response - service;

                results.total_response_time += response;
                results.total_service_time += service;
                results.total_queue_time += queue_wait;

                // this could probably all be abstracted
                let response_index = (response / step) as usize;
                while results.response_times.len() <= response_index {
                    results.response_times.push(0)
                }  
                results.response_times[response_index] += 1;
                num_completions += 1;
            }
        }

        if was_arrival {
            let size = dist.sample(&mut rng);
            let new_job = Job {
                rem_size: size,
                arrival_time: time,
            };
            next_arrival = time + arrival_dist.sample(&mut rng);
            let insertion_index = policy.insertion_index(&queue, &new_job);
            queue.insert(insertion_index, new_job);
        }
    }
    results
}

#[derive(Clone, Copy, Debug)]
enum Dist {
    Uniform(f64, f64),
    Exponential(f64),
    Hyperexponential(f64, f64, f64),
}

impl Dist {
    fn sample<R: Rng>(&self, rng: &mut R) -> f64 {
        let sample = match self {
            Dist::Uniform(low, high) => rng.random_range(*low..*high),
            Dist::Exponential(mean) => rng.sample(Exp::new(1.0 / mean).unwrap()),
            Dist::Hyperexponential(low_mean, high_mean, prob_low) => {
                let mean = if rng.random::<f64>() < *prob_low {
                    low_mean
                } else {
                    high_mean
                };
                rng.sample(Exp::new(1.0 / mean).unwrap())
            }
        };
        assert!(sample >= 0.0);
        sample
    }

    fn mean(&self) -> f64 {
        match self {
            Dist::Uniform(low, high) => (low + high) / 2.0,
            Dist::Exponential(mean) => *mean,
            Dist::Hyperexponential(low_mean, high_mean, prob_low) => {
                low_mean * prob_low + high_mean * (1.0 - prob_low)
            }
        }
    }
}

fn main() {
    let num_jobs = 2_000_000_000;
    let rho = 0.4; 
    let seed = 0;
    let step = 0.1;
    let dist = Dist::Hyperexponential(0.5, 3.0, 0.8);
    let policies = vec![
        Policy::FCFS,
    ];
    for mut policy in policies {
        let results = simulate(rho, dist, &mut policy, step, num_jobs, seed);
        assert_eq!(results.response_times.iter().sum::<usize>(), num_jobs);
        let cumulant: Vec<usize> = results
            .response_times
            .iter()
            .scan(num_jobs, |state, &count| {
                *state -= count;
                Some(*state)
            })
            .collect();
        let log_frequencies = cumulant
            .iter()
            .take((100.0/step) as usize)
            .map(|&freq| (freq as f64 / num_jobs as f64).log10());
        println!(
            "{:?};{}",
            policy,
            log_frequencies
                .map(|f| format!("{}", f))
                .collect::<Vec<String>>()
                .join(";")
        );
    }
}