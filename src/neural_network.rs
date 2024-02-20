pub mod neural_network{
    use crate::engine::engine::Value;
    use rand::distributions::{Distribution, Uniform};
    use rand::thread_rng;
    
    struct Neuron{
        weights: Vec<Value>,
        bias : Value

    }

    impl Neuron{
        pub fn new(number_of_parameters: usize) -> Self{
            let mut rng = thread_rng();
            let between = Uniform::new(-1.0,1.0);
            let mut weights = Vec::with_capacity(number_of_parameters);
            
            for _ in 0..number_of_parameters{
                weights.push(between.sample(&mut rng).into());
            }
            Neuron{
                weights,
                bias : between.sample(& mut rng).into()
            }
        }

        pub fn forward(&self, x : Vec<Value>) -> Result<Value, String>{
            if self.weights.len() != x.len() {
                Err("Vector size does not match neuron dimensions.".to_string())
            } else {
                let weights = &self.weights;
                let bias = &self.bias;
                let activation = weights.iter()
                .zip(x.iter())
                .map(|(x,&ref y)| x.clone() * y.clone())
                .sum::<Value>() + bias.clone();
                
                Ok(activation.tanh())
            }

        }
    }

    struct Layer{
        neurons : Vec<Neuron>,
        dimensionality : usize
    }

    impl Layer{
        pub fn new(dimensionality: usize, number_of_neurons:usize) -> Self{
            let mut neurons = Vec::with_capacity(number_of_neurons);
            for _ in 0..number_of_neurons{
                neurons.push(Neuron::new(dimensionality));
            }
            Layer {neurons, dimensionality}
        }

        pub fn forward(&self, x: Vec<Value>) -> Result<Vec<Value>, String>{
            if self.dimensionality != x.len() {
                Err("Vector side does not match neuron dimensions".to_string())
            } else {
                Ok(self.neurons.iter()
                .map(|neuron| neuron.forward(x.clone()).unwrap())
                .collect())
            }

        }
    }

    pub struct MultiLayerPerceptron{
        layers : Vec<Layer>,
        number_of_inputs: usize,
    }

    impl MultiLayerPerceptron{
        pub fn new(number_of_inputs: usize, neurons_allocation: Vec<usize>) -> Self {
            let mut layers = Vec::with_capacity(neurons_allocation.len());
            let layer_partition : Vec<usize> = vec!{number_of_inputs}.into_iter().chain(neurons_allocation.into_iter()).collect();
            
            let upper_bound = layer_partition.clone().len() - 1;

            for i in 0..upper_bound{
                let this_layer = Layer::new(layer_partition[i], layer_partition[i+1]);
                layers.push(this_layer);
            }

            MultiLayerPerceptron{ layers, number_of_inputs}
        }

        pub fn forward(&self, x: Vec<Value>) -> Result<Vec<Value>, String>{
            if self.number_of_inputs != x.len(){
                Err("Vector does not contain the right amount of inputs for this MLP".to_string())
            } else{
                let mut next_layer_input = x.clone();
                for layer in self.layers.iter(){
                    next_layer_input = layer.forward(next_layer_input)?;
                }
                Ok(x)
            }
            
        }
    }


}