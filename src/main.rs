 mod engine;
mod neural_network;

fn main(){
    use crate::engine::engine::Value;
    
    print!("We coded micrograd");
    let some_value = Value::new(1.0);
    let some_value2 = Value::new(-2.0);

    let some_value3 = Value::new(10.0);
    let some_value4 = Value::new(1.0/10.0);

    print!("{:?}", some_value.clone() + some_value2.clone());
    print!("{:?}", some_value3.clone() * some_value4 + some_value);
    print!("{:?}", some_value3.tanh());
}


