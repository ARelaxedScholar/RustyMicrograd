pub mod engine {
    use std::cell::RefCell;
    use std::rc::Rc;
    use std::iter::Sum;
    use core::ops::{Add, Sub, Mul, Div};
    use std::collections::{VecDeque, HashSet};
    use std::sync::atomic::{AtomicUsize, Ordering};
    
    static ID_COUNTER: AtomicUsize = AtomicUsize::new(0);

    #[derive (Clone, Debug)]
    pub struct Value{
        reference: Rc<RefCell<InnerValue>>
    }
    #[derive (Clone, Debug)]
    pub struct InnerValue {
        id : usize,
        value : f64,
        gradient : f64,
        operation: Operation
    }

    #[derive (Clone, Debug)]
    pub enum Operation {
        Addition(Value, Value),
        Multiplication(Value, Value),
        Power(Value, Exponent),
        Tanh(Value),
        Exp(Value),
        Base
    }

    #[derive (Clone, Debug)]
    pub struct Exponent{
        value : f64
    }

    impl Eq for Value{}

    impl PartialEq for Value {
        fn eq(&self, other: &Self) -> bool{
            self.reference.borrow().id == other.reference.borrow().id
        }
    }

    impl Value {
        pub fn new(value : f64) -> Self{
            Value{reference: Rc::new(
                                RefCell::new( 
                                    InnerValue {
                                        id: Self::get_initialization_id(),
                                        value, 
                                        gradient: 0.0,
                                        operation: Operation::Base
            
                                    }
                                ))
                            }
            }

        fn get_initialization_id() -> usize{
            ID_COUNTER.fetch_add(1, Ordering::Relaxed) 
        }

        pub fn backwards(&mut self){
            let mut visited_nodes = HashSet::new();
            let mut topographic_ordering = VecDeque::new();

            self.reference.borrow_mut().gradient = 1.0; //ensures the root node has its gradient initialized at 1

            
            //Then do a topological sort to order the nodes
            Self::yield_topographic_ordering(self, &mut visited_nodes, &mut topographic_ordering);
            
            //Then run the backpropagation party
            while let Some(mut value) = topographic_ordering.pop_back() {
                value.inner_backwards();
            }
        }

        fn yield_topographic_ordering(node: &Self, seen_values : &mut HashSet<usize>, current_ordering : &mut VecDeque<Self>) {
            let node_borrowed = node.reference.borrow();
            if !seen_values.contains(&node_borrowed.id){
                seen_values.insert(node_borrowed.id);
                drop(node_borrowed);

                match &node.reference.borrow().operation{
                    Operation::Addition(left_child, right_child) | Operation::Multiplication(left_child, right_child) => {
                        Self::yield_topographic_ordering(left_child, seen_values, current_ordering);
                        Self::yield_topographic_ordering(right_child, seen_values, current_ordering);
                    },
                    Operation::Power(child, _) => {
                        Self::yield_topographic_ordering(child, seen_values, current_ordering);
                    }
                    Operation::Tanh(child) | Operation::Exp(child) => Self::yield_topographic_ordering(child, seen_values, current_ordering),
                    Operation::Base => {}
                }
                current_ordering.push_back(node.clone());              
            }
        }

        fn inner_backwards(&mut self){
            let operation = self.reference.borrow_mut().operation.clone();

            match operation{
                Operation::Addition(left_child, right_child) => {
                    let mut left = left_child.reference.borrow_mut();
                    let mut right = right_child.reference.borrow_mut();

                    // By chain rule : dL/dd * dd/dc = dL/dc but since its addition dd/dc = 1.0
                    // if c = a + b, dc/da = dc/db = 1.0
                    left.gradient += 1.0 * self.reference.borrow().gradient;
                    right.gradient += 1.0 * self.reference.borrow().gradient;
                },
                Operation::Multiplication(left_child, right_child) => {
                    let mut left = left_child.reference.borrow_mut();
                    let mut right = right_child.reference.borrow_mut();

                    //Similarly, we use chain rule. 
                    //if c = a*b, dc/da = 1.0*b and dc/db = a*1.0
                    left.gradient += right.value * self.reference.borrow().gradient;
                    right.gradient += left.value * self.reference.borrow().gradient;
                },
                Operation::Power(child, exponent) => {
                    let mut child = child.reference.borrow_mut();
                    child.gradient += (exponent.value)*f64::powf(child.value, (exponent.value)-1.0);
                }
                Operation::Exp(child) => {
                    let mut child = child.reference.borrow_mut();
                    child.gradient += child.value * self.reference.borrow().gradient;
                },
                Operation::Tanh(child) => {
                    let mut child = child.reference.borrow_mut();
                    child.gradient += (1.0-child.value.powi(2)) * self.reference.borrow().gradient; 
                }
                Operation::Base => {}
            }
        }
    }

    // Activation functions
    impl Value{
        pub fn tanh(self) -> Self{
            let self_borrowed = self.reference.borrow();
            let x = self_borrowed.value.tanh();
            drop(self_borrowed);

            let squashed_self = Value::from(
                InnerValue{
                id : Self::get_initialization_id(),
                value: x,
                gradient: 0.0,
                operation: Operation::Tanh(self)});
           
            squashed_self
            }
            
    }
    // Value Operations
    pub trait Exp{
        type Output;

        fn exp(self) -> Self::Output;
    }

    pub trait Power{
        type Output;

        fn power(self, other: Exponent) -> Self::Output;
    }

    impl Sum for Value{
        fn sum<I>(iter: I) -> Self
        where
            I: Iterator<Item = Self>,
            {
                iter.fold(Self::new(0.0), |acc,x| acc + x.clone())
            }

    }

    impl From<f64> for Value{
        fn from(value: f64) -> Value{
            Value::new(value)
        }
    } 
    
    impl From<i32> for Value{
        fn from(value: i32) -> Value{
            Value::new(f64::from(value))
        }
    }

    impl From<InnerValue> for Value{
        fn from(value: InnerValue) -> Value{
            Value { reference: Rc::new(
                    RefCell::new(
                        value
                    )
                )
            }
        }
    }

    impl From<f64> for Exponent{
        fn from(value: f64) -> Exponent{
            Exponent { value}
        }
    }

    impl From<i32> for Exponent{
        fn from(value: i32) -> Exponent{
            Exponent { value: f64::from(value)}
        }
    }

    // Add Gauntlet
    impl Add for Value{
        type Output = Value;

        fn add(self, other: Self) -> Self::Output {
            Value{ reference: Rc::new(
                RefCell::new(
                InnerValue {
                    id : Self::get_initialization_id(),
                    value: self.reference.borrow().value + other.reference.borrow().value,
                    gradient: 0.0,
                    operation: Operation::Addition(self.clone(), other.clone())
                    }                   
                )
            )}
           
        }
    }
   // Multiply Gauntlet
    impl Mul for Value{
        type Output = Value;

        fn mul(self, other: Self) -> Self::Output {
            Value { reference: Rc::new(RefCell::new(
                InnerValue {
                    id : Self::get_initialization_id(),
                    value : self.reference.borrow().value * other.reference.borrow().value,
                    gradient: 0.0,
                    operation: Operation::Multiplication(self.clone(), other.clone())
            }))}
        }
    }
    impl Sub for Value{
        type Output = Value;

        fn sub(self, other: Self) -> Self::Output {
            let addititive_inverse_other = Value::from(-1) * other;
            self.add(addititive_inverse_other)           
        }
    }

    impl Div for Value{
        type Output = Value;

        fn div(self, other:Self) -> Self::Output {
            //We treat division as a special case of power rule with exponent = -1
            let exponent = Exponent::from(-1);
            let inverse_of_other = other.power(exponent);
            //Then we simply use our multiplication
            self.mul(inverse_of_other)
        }
    }
    // Exp
    impl Exp for Value{
        type Output = Value;

        fn exp(self) -> Self::Output{
            let exped_value = self.reference.borrow().value.exp();
            Value {
                reference: Rc::new(
                    RefCell::new(
                        InnerValue{
                            id : Value::get_initialization_id(),
                            value :exped_value,
                            gradient: 0.0,
                            operation: Operation::Exp(self.clone())
                        }
                    )
                )
            }
        }
    }

    impl Power for Value{
        type Output = Value;

        fn power(self, other: Exponent) -> Self::Output{
            let exponentiated_value = f64::powf(self.reference.borrow().value, f64::from(other.value));
            Value {
                reference: Rc::new(
                    RefCell::new(
                        InnerValue{
                            id : Value::get_initialization_id(),
                            value :exponentiated_value,
                            gradient: 0.0,
                            operation: Operation::Power(self.clone(), other.clone())
                        }
                    )
                )
            }
        }
    }


    
}