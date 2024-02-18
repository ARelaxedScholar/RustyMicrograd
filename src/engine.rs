pub mod engine {
    use std::cell::RefCell;
    use std::rc::Rc;
    use core::ops::{Add, Mul};
    use std::collections::{VecDeque, HashSet};
    use std::sync::atomic::{AtomicUsize, Ordering};
    
    static ID_COUNTER: AtomicUsize = AtomicUsize::new(0);

    #[derive (Clone, Debug)]
    pub struct Value {
        id : usize,
        value : f64,
        gradient : f64,
        operation: Operation
    }

    #[derive (Clone, Debug)]
    pub enum Operation {
        Addition(Rc<RefCell<Value>>, Rc<RefCell<Value>>),
        Multiplication(Rc<RefCell<Value>>, Rc<RefCell<Value>>),
        Tanh(Rc<RefCell<Value>>),
        Base
    }

    impl Eq for Value{}

    impl PartialEq for Value {
        fn eq(&self, other: &Self) -> bool{
            self.id == other.id
        }
    }

    impl Value {
        pub fn new(value : f64) -> Rc<RefCell<Self>>{
            Rc::new(RefCell::new( 
                Self {
                    id: Self::get_initialization_id(),
                    value, 
                    gradient: 0.0,
                    operation: Operation::Base
            }))
        }

        fn get_initialization_id() -> usize{
            ID_COUNTER.fetch_add(1, Ordering::Relaxed) 
        }

        pub fn backwards(root: &Rc<RefCell<Self>>){
            let mut visited_nodes = HashSet::new();
            let mut topographic_ordering = VecDeque::new();

            root.borrow_mut().gradient = 1.0; //ensures the root node has its gradient initialized at 1

            
            //Then do a topological sort to order the nodes
            Self::yield_topographic_ordering(root, &mut visited_nodes, &mut topographic_ordering);
            
            //Then run the backpropagation party
            while let Some(value) = topographic_ordering.pop_front() {
                value.inner_backwards();
            }
        }

        fn yield_topographic_ordering(node: &Rc<RefCell<Self>>, seen_values : &mut HashSet<usize>, current_ordering : &mut VecDeque<Rc<RefCell<Self>>>) {
            let node_borrowed = node.borrow();
            if !seen_values.contains(&node_borrowed.id){
                seen_values.insert(node_borrowed.id);
                drop(node_borrowed);

                match &node.borrow().operation{
                    Operation::Addition(left_child, right_child) | Operation::Multiplication(left_child, right_child) => {
                        Self::yield_topographic_ordering(left_child, seen_values, current_ordering);
                        Self::yield_topographic_ordering(right_child, seen_values, current_ordering);
                    },
                    Operation::Tanh(child) => Self::yield_topographic_ordering(child, seen_values, current_ordering),
                    Operation::Base => {}
                }
                current_ordering.push_back(Rc::clone(node));              
            }
        }

        fn inner_backwards(&mut self){
            match self.operation{
                Operation::Addition(ref mut left_child, ref mut right_child) => {
                    let mut left = left_child.borrow_mut();
                    let mut right = right_child.borrow_mut();

                    // By chain rule : dL/dd * dd/dc = dL/dc but since its addition dd/dc = 1.0
                    // if c = a + b, dc/da = dc/db = 1.0
                    left.gradient += 1.0 * self.gradient;
                    right.gradient += 1.0 * self.gradient;
                },
                Operation::Multiplication(ref mut left_child, ref mut right_child) => {
                    let mut left = left_child.borrow_mut();
                    let mut right = right_child.borrow_mut();

                    //Similarly, we use chain rule. 
                    //if c = a*b, dc/da = 1.0*b and dc/db = a*1.0
                    left.gradient += right_child.value * self.gradient;
                    right.gradient += left_child.value * self.gradient;
                },
                Operation::Tanh(ref mut child) => {
                    child.gradient = (1.0-child.value.powi(2)) * self.gradient; 
                }
                Operation::Base => {}
            }
        }
    }

    // Activation functions
    impl Value{
        pub fn tanh_using_builtin(self) -> Self{
            Value{
                id : Self::get_initialization_id(),
                value: self.value.tanh(),
                gradient: 0.0,
                operation: Operation::Tanh(Rc::new(RefCell::new(self)))
            }
        }
    }
    // Value Operations
    impl From<f64> for Rc<RefCell<Value>>{
        fn from(value: f64) -> Rc<RefCell<Value>>{
            Value::new(value)
        }
    } 
    
    impl From<i32> for Rc<RefCell<Value>>{
        fn from(value: i32) -> Rc<RefCell<Value>>{
            Value::new(f64::from(value))
        }
    }
    // Add Gauntlet
    impl<T: Into<Rc<RefCell<Value>>>> Add<T> for Rc<RefCell<Value>>{
        type Output = Rc<RefCell<Value>>;

        fn add(self, other: T) -> Self::Output {
            let innered = other.into();
            Value {
                id : Self::get_initialization_id(),
                value: self.borrow().value + innered.borrow().value,
                gradient: 0.0,
                operation: Operation::Addition(self.clone(), innered.clone())
            }
        }
    }
   // Multiply Gauntlet
    impl<T: Into<Rc<RefCell<Value>>>> Mul<T> for Rc<RefCell<Value>>{
        type Output = Rc<RefCell<Value>>;

        fn mul(self, other: T) -> Self::Output {
            let innered = other.into();
            Rc::new(RefCell::new(Value {
                id : Self::get_initialization_id(),
                value : self.borrow().value * other.borrow().value,
                gradient: 0.0,
                operation: Operation::Multiplication(self.clone(), other.clone())
            }))
        }
    }
}

