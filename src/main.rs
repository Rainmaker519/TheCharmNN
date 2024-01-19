use std::f64::consts;

#[derive(Copy, Clone, PartialEq)]
enum ActivationType {
    Sigmoid,
    Tanh,
    Linear,
}

type PreActivationValue = (f64,ActivationType);

trait PAVForwardPass {
    fn activate(&self) -> Option<(f64,f64)>;
}
impl PAVForwardPass for PreActivationValue {
    fn activate(&self) -> Option<(f64, f64)> {
        return if self.1.clone() == ActivationType::Linear { //0 is linear
            Option::Some((self.0,1.0))
        } else if self.1.clone() == ActivationType::Sigmoid { //1 is sigmoid
            let post_activation = 1.0 / (1.0 + consts::E.powf(-self.0));
            let post_prime_activation = consts::E.powf(self.0)
                / (1.0 + consts::E.powf(self.0)).powf(2.0);
            Option::Some((post_activation,post_prime_activation))
        } else {
            Option::None
        }
    }
}

type Node = (PreActivationValue, Vec<Weight>, Vec<Weight>, usize, usize); // last 2 next and prev layer sizes

trait NodeForwardPass {
    fn get_nv_pairs(&self) -> Vec<(usize,(f64,f64))>;
    fn set_preactivation_value(&mut self, v: f64, t: ActivationType);
}
impl NodeForwardPass for Node {
    fn get_nv_pairs(&self)  -> Vec<(usize,(f64,f64))> {
        let mut result: Vec<(usize,(f64,f64))> = vec![];
        for i in 0..self.3 { // size of next layer
            if self.1[i].is_some() {
                let post_act_val = &self.0.activate();
                match post_act_val {
                    None => {},
                    Some(x) => {result.push((i,*x))},
                }
            }
        }
        result
    }
    fn set_preactivation_value(&mut self, v: f64, t: ActivationType) {
        self.0.0 = v;
        self.0.1 = t;
    }
}

type Layer = (Vec<Node>, usize);
trait MakeLayer {
    fn make(v: Vec<Node>) -> Layer;
}
impl MakeLayer for Layer {
    fn make(v: Vec<Node>) -> Layer {
        let l = *&v.len().clone();
        (v, l)
    }
}

type Weight = Option<f64>;

trait MakeWeight {
    fn make(v: f64) -> Weight;
}
impl MakeWeight for Weight {
    fn make(v: f64) -> Weight {
        if v > 0f64 {
            return Option::Some(v);
        }
        Option::None
    }
}

type Network = (Vec<Layer>,Vec<usize>);


fn main() {
    let v1: PreActivationValue = (0.0443, ActivationType::Sigmoid);

    let n1: Node = (v1,
         vec![Weight::make(0.12), Weight::make(0.16), Weight::make(0f64)],
         vec![Weight::make(0f64), Weight::make(0.0334), Weight::make(0.432)],
        3, 3,
    );

    let nv_pairs = n1.get_nv_pairs();

    for i in nv_pairs.iter() {
        dbg!(i.0, i.1.0, i.1.1);
        println!();
    }

    println!("Hi test commit!");
}








