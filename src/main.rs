 mod engine;
mod neural_network;

fn main(){
    use crate::engine::engine::Value;
    let val1 = Value::new(0.0); //0.0
    let val2 = Value::new(3.0); //3.0
    let val3 = Value::from(15); //15.0

    let val4 = val1.clone() + val2.clone(); //3.0
    let val5 = val2.clone() * val3.clone(); //45.0

    let val6 = Value::from(10) * val4.clone(); //30.0

    let mut val7 = val5.clone() + val6.clone();

    println!("{:?}", val7.clone());
    println!();
    val7.backwards();
    println!("{:?}", val7);
}