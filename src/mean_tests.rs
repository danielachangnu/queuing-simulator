use super::*;
#[test]
fn test_dist_mean() {
    let uniform = Dist::Uniform(2.0, 4.0);
    assert!((uniform.mean() - 3.0).abs() < EPSILON);

    let exp = Dist::Exponential(5.0);
    assert!((exp.mean() - 5.0).abs() < EPSILON);

    let hyper = Dist::Hyperexponential(1.0, 10.0, 0.8);
    assert!((hyper.mean() - 2.8).abs() < EPSILON);
}
#[test]
fn test_policy_fcfs() {
    let mut queue = vec![
        Job {
            rem_size: 1.0,
            arrival_time: 1.0,
            original_size: 1.0,
        },
        Job {
            rem_size: 1.0,
            arrival_time: 5.0,
            original_size: 1.0,
        },
    ];
    let new_job = Job {
        rem_size: 1.0,
        arrival_time: 6.0,
        original_size: 1.0,
    };

    let policy = Policy::FCFS;

    assert_eq!(policy.insertion_index(&queue, &new_job).unwrap(), 2);

    policy.reorder(&mut queue);
    assert_eq!(queue[0].arrival_time, 1.0);
    assert_eq!(queue[1].arrival_time, 5.0);
}

#[test]
fn test_bva_queue_empty() {
    let mut queue: Vec<Job> = vec![];
    let new_job = Job {
        rem_size: 1.0,
        arrival_time: 0.0,
        original_size: 1.0,
    };

    assert_eq!(Policy::FCFS.insertion_index(&queue, &new_job).unwrap(), 0);

    Policy::PLCFS.reorder(&mut queue);
    assert_eq!(queue.len(), 0);
}

#[test]
fn test_bva_queue_single_item() {
    let mut queue: Vec<Job> = vec![Job {
        rem_size: 1.0,
        arrival_time: 5.0,
        original_size: 1.0,
    }];

    Policy::PLCFS.reorder(&mut queue);
    assert_eq!(queue[0].arrival_time, 5.0);
}

#[test]
fn test_bva_plcfs_exact_arrival_ties() {
    let mut queue = vec![
        Job {
            rem_size: 1.0,
            arrival_time: 5.0,
            original_size: 1.0,
        },
        Job {
            rem_size: 2.0,
            arrival_time: 5.0,
            original_size: 2.0,
        },
        Job {
            rem_size: 3.0,
            arrival_time: 2.0,
            original_size: 3.0,
        },
    ];

    Policy::PLCFS.reorder(&mut queue);

    assert_eq!(queue[2].arrival_time, 2.0);

    assert_eq!(queue[0].rem_size, 1.0);
    assert_eq!(queue[1].rem_size, 2.0);
}

#[test]
fn test_system_littles_law() {
    let lambda = 0.6;
    let config = Config {
        debug: false,
        response_time_histogram: None,
    };

    let mut results = simulate(
        lambda,
        Dist::Exponential(1.0),
        Policy::FCFS,
        10_000,
        123,
        &config,
    );

    let mean_response = results.mean_response_time();
    let mean_jobs = results.mean_number_of_jobs(lambda);

    let expected_jobs = lambda * mean_response;

    assert!((mean_jobs - expected_jobs).abs() < EPSILON);
}

#[test]
fn test_system_littles_law2() {
    let theoretical_lambda = 0.6;
    let config = Config {
        debug: false,
        response_time_histogram: None,
    };

    let mut results = simulate(
        theoretical_lambda,
        Dist::Exponential(1.0),
        Policy::FCFS,
        50_000,
        123,
        &config,
    );

    let expected_l = results.mean_number_of_jobs(theoretical_lambda);

    let actual_l = results.total_response_time / results.total_time;

    let margin = actual_l * 0.02;

    assert!((actual_l - expected_l).abs() < margin);
}
