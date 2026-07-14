use noisy_float::prelude::*;
use rand::prelude::*;
use rand_distr::{Beta, ChiSquared, Exp, InverseGaussian, Normal, Pareto};

use std::f64::INFINITY;
const EPSILON: f64 = 1e-8;
use std::f64::consts::PI;

struct Job {
    rem_size: f64,
    arrival_time: f64,
    original_size: f64,
}
struct Config {
    response_time_histogram: Option<f64>,
    debug: bool,
}

//Only insertion policies
#[derive(Debug, Copy, Clone)]
enum Policy {
    FCFS,
    PLCFS,
    LCFS,
    SRPT,
    PSJF,
}
impl Policy {
    fn insertion_index(&self, queue: &Vec<Job>, new_job: &Job) -> Option<usize> {
        match self {
            Policy::FCFS => Some(queue.len()),
            Policy::PLCFS => Some(0),
            Policy::LCFS => {
                if queue.is_empty() {
                    Some(0)
                } else {
                    Some(1)
                }
            }
            Policy::SRPT => None,
            Policy::PSJF => None,
        }
    }

    fn reorder(&self, queue: &mut Vec<Job>) {
        match self {
            //srpt-improve-or for changes to this
            Policy::FCFS => (),
            Policy::PLCFS => (),
            Policy::LCFS => (),
            Policy::SRPT => queue.sort_by_key(|job| n64(job.rem_size)),
            Policy::PSJF => queue.sort_by_key(|job| n64(job.original_size)),
        }
    }
}

struct Results {
    response_times: Vec<usize>,
    step: Option<f64>,
    total_response_time: f64,
    total_queue_time: f64,
    num_jobs: usize,
    total_service_time: f64,
    mean_response_time: f64,
    mean_queue_time: f64,
    mean_service_time: f64,
    mean_number_of_jobs: f64,
    total_time: f64,
}

impl Results {
    pub fn mean_response_time(&mut self) -> f64 {
        self.mean_response_time = self.total_response_time / (self.num_jobs as f64);
        self.mean_response_time
    }

    pub fn mean_queue_time(&mut self) -> f64 {
        self.mean_queue_time = self.total_queue_time / (self.num_jobs as f64);
        self.mean_queue_time
    }

    pub fn mean_service_time(&mut self) -> f64 {
        self.mean_service_time = self.total_service_time / (self.num_jobs as f64);
        self.mean_service_time
    }

    pub fn mean_number_of_jobs(&mut self, lambda: f64) -> f64 {
        // little's law
        self.mean_number_of_jobs = lambda * self.mean_response_time() as f64;
        self.mean_number_of_jobs
    }

    pub fn update_response_time_histogram(&mut self, response: f64) {
        if let Some(step) = self.step {
            let response_index = (response / step) as usize;
            while self.response_times.len() <= response_index {
                self.response_times.push(0)
            }
            self.response_times[response_index] += 1;
        }
    }
}

fn simulate(
    lambda: f64,
    dist: Dist,
    policy: Policy, // shouldn't be mutable for now
    num_jobs: usize,
    seed: u64,
    config: &Config,
) -> Results {
    assert!((dist.mean() - 1.0).abs() < EPSILON); // mean = lambda is like the load, so if the mean is around 1 its accurate. if not, its not accurate
    let mut rng = StdRng::seed_from_u64(seed);
    let mut time = 0.0; // easy way to track response time
    let arrival_dist = Exp::new(lambda).unwrap();
    let mut next_arrival = rng.sample(arrival_dist);
    let mut num_completions = 0;
    let mut queue: Vec<Job> = vec![];
    let mut results = Results {
        step: config.response_time_histogram,
        response_times: vec![],
        total_response_time: 0.0,
        total_queue_time: 0.0,
        num_jobs,
        total_service_time: 0.0,
        mean_response_time: 0.0,
        mean_queue_time: 0.0,
        mean_service_time: 0.0,
        mean_number_of_jobs: 0.0,
        total_time: 0.0,
    };

    if config.debug {
        println!("lambda: {lambda}");
    }

    while num_completions < num_jobs {
        if config.debug {
            println!("time: {time} ");
            println!("next arrival time: {next_arrival}");
            std::io::stdin()
                .read_line(&mut String::new())
                .expect("continued");
        }
        let next_event_diff =
            (next_arrival - time).min(queue.first().map_or(INFINITY, |j| j.rem_size)); // pick whichever one will happen next, either the next arrival or the current job is finished
        let was_arrival = next_event_diff == (next_arrival - time); // if the next event is an arrival make sure to update the flag accordingly
        time += next_event_diff;
        if !queue.is_empty() {
            let job = queue.first_mut().unwrap();
            job.rem_size -= next_event_diff;
            assert!(job.rem_size >= -EPSILON);
            if job.rem_size <= EPSILON {
                let job = queue.remove(0);
                let response = time - job.arrival_time;
                let service = job.original_size;
                let wait_time = response - service;

                results.total_response_time += response;
                results.total_service_time += service;
                results.total_queue_time += wait_time;

                if let Some(step) = config.response_time_histogram {
                    results.update_response_time_histogram(response);
                }
                num_completions += 1;
            }
        }

        if was_arrival {
            let size = dist.sample(&mut rng);
            let new_job = Job {
                rem_size: size,
                arrival_time: time,
                original_size: size,
            };
            next_arrival = time + arrival_dist.sample(&mut rng);
            let insertion_index = policy.insertion_index(&queue, &new_job);
            match policy.insertion_index(&queue, &new_job) {
                Some(index) => {
                    queue.insert(index, new_job);
                }
                None => {
                    queue.push(new_job);
                    policy.reorder(&mut queue);
                }
            }
        }
    }

    assert!(results.total_queue_time >= -EPSILON);
    assert!(results.total_response_time >= results.total_service_time - EPSILON);
    results.total_time = time;
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
    let num_jobs = 20;
    let rho = 0.4;
    let seed = 0;
    let dist = Dist::Hyperexponential(0.5, 3.0, 0.8);
    let policies = vec![Policy::FCFS, Policy::PLCFS];
    let mut inputOption = String::new();
    std::io::stdin()
        .read_line(&mut inputOption)
        .expect("could not read line");

    let choice = inputOption.trim().parse().unwrap_or(0);

    let config = match choice {
        1 => Config {
            debug: false,
            response_time_histogram: Some(0.0001),
        },
        2 => Config {
            debug: true,
            response_time_histogram: None,
        },
        3 => Config {
            debug: true,
            response_time_histogram: Some(0.0001),
        },
        _ => Config {
            debug: false,
            response_time_histogram: None,
        },
    };

    for policy in policies {
        let results = simulate(rho, dist, policy, num_jobs, seed, &config);

        if let Some(step) = config.response_time_histogram {
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
                .take((100.0 / step) as usize)
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
}

#[cfg(test)]
mod mean_tests;
