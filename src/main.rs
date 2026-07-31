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

#[derive(Debug, Copy, Clone)]
enum Policy {
    FCFS,
    PLCFS,
    LCFS,
    SRPT,
    PSJF,
    PS,
    LAS,
    LRPT,
}
impl Policy {
    fn work(&self, running: &mut Vec<Job>) -> f64 {
        match self {
            Policy::PS | Policy::LRPT | Policy::LAS => {
                if running.is_empty() {
                    0.0
                } else {
                    1.0 / running.len() as f64
                }
            }

            _ => {
                if running.is_empty() {
                    0.0
                } else {
                    1.0
                }
            }
        }
    }

    fn arrival(&self, waiting: &mut Vec<Job>, running: &mut Vec<Job>, new_job: Job) {
        match self {
            Policy::FCFS => {
                if running.is_empty() {
                    running.push(new_job);
                } else {
                    waiting.push(new_job);
                }
            }

            Policy::PLCFS => {
                if !running.is_empty() {
                    waiting.insert(0, running.remove(0));
                }
                running.push(new_job);
            }

            Policy::LCFS => {
                if running.is_empty() {
                    running.push(new_job);
                } else {
                    waiting.insert(0, new_job);
                }
            }

            Policy::SRPT => {
                if running.is_empty() {
                    running.push(new_job);
                } else if new_job.rem_size < running[0].rem_size {
                    waiting.push(running.remove(0));
                    waiting.sort_by_key(|job| n64(job.rem_size));
                    running.push(new_job);
                } else {
                    waiting.push(new_job);
                    waiting.sort_by_key(|job| n64(job.rem_size));
                }
            }

            Policy::PSJF => {
                if running.is_empty() {
                    running.push(new_job);
                } else if new_job.original_size < running[0].original_size {
                    waiting.push(running.remove(0));
                    waiting.sort_by_key(|job| n64(job.original_size));
                    running.push(new_job);
                } else {
                    waiting.push(new_job);
                    waiting.sort_by_key(|job| n64(job.original_size));
                }
            }

            Policy::PS => {
                running.push(new_job);
            }

            Policy::LAS => {
                if running.is_empty() {
                    running.push(new_job);
                } else {
                    let current_attained = running[0].original_size - running[0].rem_size;
                    let new_attained = 0.0;
                    if new_attained < current_attained {
                        waiting.append(running);
                        running.push(new_job);
                        waiting.sort_by_key(|job| n64(job.original_size - job.rem_size));
                    } else {
                        running.push(new_job);
                    }
                }
            }

            Policy::LRPT => {
                if running.is_empty() {
                    running.push(new_job);
                } else {
                    let current_rem = running[0].rem_size;
                    if new_job.rem_size > current_rem {
                        waiting.append(running);
                        running.push(new_job);
                        waiting.sort_by_key(|job| -n64(job.rem_size));
                    } else if new_job.rem_size == current_rem {
                        running.push(new_job);
                    } else {
                        waiting.push(new_job);
                        waiting.sort_by_key(|job| -n64(job.rem_size));
                    }
                }
            }
        }
    }

    fn completion(&self, waiting: &mut Vec<Job>, running: &mut Vec<Job>) {
        match self {
            Policy::PS => (),
            Policy::LAS => {
                if running.is_empty() && !waiting.is_empty() {
                    let first = waiting.remove(0);
                    let target_attained = first.original_size - first.rem_size;
                    running.push(first);

                    while !waiting.is_empty() {
                        let next_attained = waiting[0].original_size - waiting[0].rem_size;
                        if (next_attained - target_attained).abs() <= EPSILON {
                            running.push(waiting.remove(0));
                        } else {
                            break;
                        }
                    }
                }
            }

            Policy::LRPT => {
                if running.is_empty() && !waiting.is_empty() {
                    let first = waiting.remove(0);
                    let target_rem = first.rem_size;
                    running.push(first);

                    while !waiting.is_empty() {
                        if (waiting[0].rem_size - target_rem).abs() <= EPSILON {
                            running.push(waiting.remove(0));
                        } else {
                            break;
                        }
                    }
                }
            }

            _ => {
                if !waiting.is_empty() {
                    running.push(waiting.remove(0));
                }
            }
        }
    }

    fn time_to_preemption(&self, waiting: &[Job], running: &[Job], work_rate: f64) -> f64 {
        match self {
            Policy::LAS if !running.is_empty() && !waiting.is_empty() => {
                let current_attained = running[0].original_size - running[0].rem_size;
                let next_attained = waiting[0].original_size - waiting[0].rem_size;

                ((next_attained - current_attained) / work_rate).max(0.0)
            }
            Policy::LRPT if !running.is_empty() && !waiting.is_empty() => {
                let current_rem = running[0].rem_size;
                let next_rem = waiting[0].rem_size;

                ((current_rem - next_rem) / work_rate).max(0.0)
            }
            _ => INFINITY,
        }
    }

    fn handle_preemption(&self, waiting: &mut Vec<Job>, running: &mut Vec<Job>) {
        if waiting.is_empty() || running.is_empty() {
            return;
        }

        match self {
            Policy::LAS => {
                let current_attained = running[0].original_size - running[0].rem_size;

                while !waiting.is_empty() {
                    let next_attained = waiting[0].original_size - waiting[0].rem_size;
                    if (next_attained - current_attained).abs() <= EPSILON {
                        running.push(waiting.remove(0));
                    } else {
                        break;
                    }
                }
            }
            Policy::LRPT => {
                let current_rem = running[0].rem_size;

                while !waiting.is_empty() {
                    if (waiting[0].rem_size - current_rem).abs() <= EPSILON {
                        running.push(waiting.remove(0));
                    } else {
                        break;
                    }
                }
            }
            _ => (),
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
    total_slowdown: f64,
    mean_response_time: f64,
    mean_queue_time: f64,
    mean_service_time: f64,
    mean_number_of_jobs: f64,
    mean_slowdown: f64,
    total_time: f64,
}

impl Results {
    pub fn mean_response_time(&mut self) {
        self.mean_response_time = self.total_response_time / (self.num_jobs as f64);
    }

    pub fn mean_queue_time(&mut self) {
        self.mean_queue_time = self.total_queue_time / (self.num_jobs as f64);
    }

    pub fn mean_service_time(&mut self) {
        self.mean_service_time = self.total_service_time / (self.num_jobs as f64);
    }

    pub fn mean_number_of_jobs(&mut self, lambda: f64) {
        // little's law
        self.mean_number_of_jobs = lambda * self.mean_response_time as f64;
    }

    pub fn mean_slowdown(&mut self) {
        self.mean_slowdown = self.total_slowdown / (self.num_jobs as f64);
    }

    pub fn update_response_time_histogram(&mut self, response: f64) {
        if let Some(step) = self.step {
            let response_index = (response / step) as usize;
            while self.response_times.len() <= response_index {
                self.response_times.push(0);
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
    let mut num_arrivals = 0;
    let mut jobs_on_arrival = 0;
    let mut jobs_waiting: Vec<Job> = vec![];
    let mut jobs_in_progress: Vec<Job> = vec![];
    let mut results = Results {
        step: config.response_time_histogram,
        response_times: vec![],
        total_response_time: 0.0,
        total_queue_time: 0.0,
        num_jobs,
        total_service_time: 0.0,
        mean_response_time: 0.0,
        total_slowdown: 0.0,
        mean_queue_time: 0.0,
        mean_service_time: 0.0,
        mean_number_of_jobs: 0.0,
        mean_slowdown: 0.0,
        total_time: 0.0,
    };

    if config.debug {
        println!("lambda: {lambda}");
    }

    while num_completions < num_jobs {
        let mut work_rate = policy.work(&mut jobs_in_progress);
        if config.debug {
            println!("time: {time} ");
            println!("next arrival time: {next_arrival}");
            std::io::stdin()
                .read_line(&mut String::new())
                .expect("continued");
        }

        let mut time_to_completion = INFINITY;
        for job in &jobs_in_progress {
            let time_for_this_job = job.rem_size / work_rate;

            if time_for_this_job < time_to_completion {
                time_to_completion = time_for_this_job;
            }
        }
        let time_to_preemption = policy.time_to_preemption(&jobs_waiting, &jobs_in_progress, work_rate);
        
        let next_event_diff = (next_arrival - time).min(time_to_completion).min(time_to_preemption);
        let was_arrival = next_event_diff == (next_arrival - time);
        
        let was_preemption = next_event_diff == time_to_preemption 
            && !was_arrival 
            && next_event_diff < time_to_completion;
        

        let next_event_diff = (next_arrival - time).min(time_to_completion);
        let was_arrival = next_event_diff == (next_arrival - time); 
        time += next_event_diff;
        for job in &mut jobs_in_progress {
            job.rem_size -= next_event_diff * work_rate;
            assert!(job.rem_size >= -EPSILON);
        }

        let mut i = 0;
        while i < jobs_in_progress.len() {
            if jobs_in_progress[i].rem_size <= EPSILON {
                let job = jobs_in_progress.remove(i);

                let response = time - job.arrival_time;
                let service = job.original_size;
                let wait_time = response - service;
                let slowdown = response/service;

                results.total_response_time += response;
                results.total_service_time += service;
                results.total_queue_time += wait_time;
                results.total_slowdown+= slowdown;

                if let Some(step) = config.response_time_histogram {
                    results.update_response_time_histogram(response);
                }
                num_completions += 1;

                policy.completion(&mut jobs_waiting, &mut jobs_in_progress);
            } else {
                i += 1;
            }
        }

        if was_preemption {
            policy.handle_preemption(&mut jobs_waiting, &mut jobs_in_progress);
        }

        if was_arrival {
            let size = dist.sample(&mut rng);
            let new_job = Job {
                rem_size: size,
                arrival_time: time,
                original_size: size,
            };
            next_arrival = time + arrival_dist.sample(&mut rng);
            num_arrivals += 1;
            jobs_on_arrival += jobs_waiting.len() + jobs_in_progress.len();
            policy.arrival(&mut jobs_waiting, &mut jobs_in_progress, new_job);
        }
    }

    assert!(results.total_queue_time >= -EPSILON);
    assert!(results.total_response_time >= results.total_service_time - EPSILON);
    results.total_time = time;
    let mean_number_of_jobs_on_arrival = jobs_on_arrival as f64 / num_arrivals as f64;
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
    let num_jobs = 100_000_000;
    let rho = 0.4;
    let seed = 10;
    let dist = Dist::Hyperexponential(0.5, 3.0, 0.8);
    let policies = vec![Policy::LCFS];

    /*
    let mut inputOption = String::new();
    std::io::stdin()
        .read_line(&mut inputOption)
        .expect("could not read line");

    let choice = inputOption.trim().parse().unwrap_or(0);*/

    let choice = std::env::args()
        .nth(1)
        .unwrap_or("0".to_string())
        .parse()
        .unwrap_or(0);

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
        println!("{}", results.mean_response_time);
    }
}

#[cfg(test)]
mod mean_tests;