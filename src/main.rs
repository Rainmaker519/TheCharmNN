use std::f64::consts;
use std::fs;

#[derive(Copy, Clone, PartialEq, Debug)]
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

type Node = (PreActivationValue, Vec<Weight>, Vec<Weight>, usize, usize, f64); // last 2 next and prev layer sizes + bias

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

trait NetworkBuild {
    fn start_network(input_layer_size: usize) -> Network;
    fn add_layer(&mut self, l_chunk: ((Vec<Vec<Weight>>,Vec<f64>),(Vec<Vec<Weight>>,Vec<f64>)));
}
impl NetworkBuild for Network {
    fn start_network(input_layer_size: usize) -> Network {
        let input_layer: Layer = (vec![], input_layer_size);
        let network: Network = (vec![input_layer], vec![input_layer_size]);
        network
    }
    fn add_layer(&mut self, l_chunks: ((Vec<Vec<Weight>>, Vec<f64>),(Vec<Vec<Weight>>,Vec<f64>))) {
        let weights_0 = l_chunks.0.0;
        let weights_1 = l_chunks.1.0;
        let biases = l_chunks.0.1;
        let mut nodes: Vec<Node> = vec![];
        let mut in_node_w_holder = vec![];

        for i in 0..weights_1.len() {
            let mut in_bank = vec![];
            for w in weights_0.iter() {
                in_bank.push(w[i]);
            }
            in_node_w_holder.push(in_bank);
        }

        for i in 0..weights_1.len() {
            nodes.push((
                (0f64,ActivationType::Sigmoid),
                in_node_w_holder.clone()[i].clone(),
                weights_1[i].clone(),
                weights_0.len(),
                weights_1.len(),
                biases[i]
            ));
        }

        self.0.push((nodes, weights_1.len()));
        //checked and weights connected correctly according to test in main
    }
}


fn main() {
    let v1: PreActivationValue = (0.0443, ActivationType::Sigmoid);

    let n1: Node = (v1,
         vec![Weight::make(0.12), Weight::make(0.16), Weight::make(0f64)],
         vec![Weight::make(0f64), Weight::make(0.0334), Weight::make(0.432)],
        3, 3, 1.3
    );

    let nv_pairs = n1.get_nv_pairs();

    for i in nv_pairs.iter() {
        dbg!(i.0, i.1.0, i.1.1);
        println!();
    }

    //let file: &str = "test_network.txt";
    //let file = fs::read_to_string(file);

    let layer_sizes: Vec<usize> = vec![4,2,4,5];//read from file when fr

    let mut layers = Network::start_network(layer_sizes[0]);

    let mut w_chunks = vec![
        vec![Weight::make(0.2), Weight::make(0.4)],
        vec![Weight::make(0.3), Weight::make(0.3)],
        vec![Weight::make(0.2), Weight::make(0.4)],
        vec![Weight::make(0.3), Weight::make(0.3)],
    ];
    let mut bias_chunk = vec![1.2,-0.83];
    let l_chunk: (Vec<Vec<Weight>>,Vec<f64>) = (w_chunks,bias_chunk);

    let mut w_chunks1 = vec![
        vec![Weight::make(0.2), Weight::make(0.4),Weight::make(0.3), Weight::make(0.3)],
        vec![Weight::make(0.3), Weight::make(0.3),Weight::make(0.2), Weight::make(0.4)],
    ];
    let mut bias_chunk1 = vec![1.2,-0.83,1.2,-0.83];
    let l_chunk1: (Vec<Vec<Weight>>,Vec<f64>) = (w_chunks1,bias_chunk1);

    layers.add_layer((l_chunk,l_chunk1));
    dbg!(&layers.0[1].0[0]);

    //add_layer takes two l_chunks
    //an l_chunk represents the weights from the prev layers nodes, the first ind being the node in the prev layer
    //  and the second ind being the node in this layer/
    //in addition an l_chunk has the bias of the layer, the first of the two passed will be used for
    //  the layer being added to the network


    println!("Hi test commit!");
}








