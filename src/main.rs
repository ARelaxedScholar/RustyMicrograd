mod engine;
mod neural_network;


fn main(){
    use crate::engine::engine::Value;
    use crate::neural_network::neural_network::MultiLayerPerceptron;
    //A simple neural network
    let a_simple_network = MultiLayerPerceptron::new(3, vec![4, 7, 5, 1]);
    let sample_vector = vec![Value::from(3), Value::from(-9.5), Value::from(4.5)];

    for i in 0..10{
        let some_value = a_simple_network.forward(sample_vector.clone()).unwrap();
        println!("Value on try {}: {:?}", i, some_value);
    }
    
}