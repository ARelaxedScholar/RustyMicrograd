pub mod neural_network{
    use rand::distributions::{Distribution, Uniform};
    use rand::thread_rng;
    struct Neuron{
        weights: Vec<f64>,
        bias : f64

    }

    impl Neuron{
        pub fn new(number_of_parameters: usize) -> Self{
            let mut rng = thread_rng();
            let between = Uniform::new(-1.0,1.0);
            let mut weights = Vec::with_capacity(number_of_parameters);
            
            for _ in 0..number_of_parameters{
                weights.push(between.sample(&mut rng));
            }
            Neuron{
                weights,
                bias : between.sample(& mut rng)
            }
        }
    }
}