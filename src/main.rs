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

type Node<const NextLayerSize: usize, const PrevLayerSize: usize> =
    (PreActivationValue, [bool; NextLayerSize], [bool; PrevLayerSize]);

trait NodeForwardPass {
    fn get_nv_pairs<const NF: usize, const NP: usize>(&self) -> Vec<(usize,(f64,f64))>;
}
impl<const NF: usize, const NP: usize> NodeForwardPass for Node<NF,NP> {
    fn get_nv_pairs<const InnerNF: usize, const InnerNP: usize>(&self)  -> Vec<(usize,(f64,f64))> {
        let mut result: Vec<(usize,(f64,f64))> = vec![];
        for i in 0..NF {
            if self.1[i].clone() {
                let post_act_val = &self.0.activate();
                match post_act_val {
                    None => {},
                    Some(x) => {result.push((i,*x))},
                }
            }
        }
        result
    }
}


fn main() {
    let v1: PreActivationValue = (0.0443, ActivationType::Sigmoid);

    let n1: Node<3,3> = (v1, [true, false, true], [false, true, true]);

    const P: usize = 1;

    let nv_pairs = n1.get_nv_pairs::<P,3>();

    for i in nv_pairs.iter() {
        dbg!(i.0, i.1.0, i.1.1);
        println!();
    }

    println!("Hi test commit!");
}








