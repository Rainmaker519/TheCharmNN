use std::f64::consts;
use std::fs;
use std::str::FromStr;

#[derive(Copy, Clone, PartialEq, Debug)]
enum ActivationType {
    Sigmoid,
    Tanh,
    Linear,
}

type PreActivationValue = (f64,ActivationType);

trait PAVUtils {
    fn activate(&self) -> Option<(f64,f64)>;
}
impl PAVUtils for PreActivationValue {
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

trait NodeUtils {
    fn get_nv_pairs(&self) -> Vec<(usize,(f64,f64))>;
    fn set_preactivation_value(&mut self, v: f64, t: ActivationType);
}
impl NodeUtils for Node {
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
trait LayerUtils {
    fn make(v: Vec<Node>) -> Layer;
}
impl LayerUtils for Layer {
    fn make(v: Vec<Node>) -> Layer {
        let l = *&v.len().clone();
        (v, l)
    }
}

type Weight = Option<f64>;

trait WeightUtils {
    fn make(v: f64) -> Weight;
}
impl WeightUtils for Weight {
    fn make(v: f64) -> Weight {
        if v > 0f64 {
            return Option::Some(v);
        }
        Option::None
    }
}

type Network = (Vec<Layer>,Vec<usize>);

trait NetworkUtils {
    fn start_network(input_layer_size: usize) -> Network;
    fn add_hidden_layer(&mut self, l_chunk: ((Vec<Vec<Weight>>, Vec<f64>), (Vec<Vec<Weight>>, Vec<f64>)));
    fn add_output_layer(&mut self, l_chunk: (Vec<Vec<Weight>>, Vec<f64>));

    //fn forward_pass();
}
impl NetworkUtils for Network {
    fn start_network(input_layer_size: usize) -> Network {
        let mut input_nodes: Vec<Node> = vec![];
        for i in 0..input_layer_size {
            input_nodes.push(((0.0f64, ActivationType::Linear), vec![], vec![], 0, 0, 0f64));
        }
        let input_layer: Layer = (input_nodes, input_layer_size);
        let network: Network = (vec![input_layer], vec![input_layer_size]);
        network
    }
    fn add_hidden_layer(&mut self, l_chunks: ((Vec<Vec<Weight>>, Vec<f64>), (Vec<Vec<Weight>>, Vec<f64>))) {
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
                (0f64, ActivationType::Sigmoid),
                weights_1[i].clone(),
                in_node_w_holder.clone()[i].clone(),
                weights_1[0].len(),
                weights_0.len(),
                biases[i]
            ));
        }

        self.0.push((nodes, weights_1.len()));
        //checked and weights connected correctly according to test in main
    }

    fn add_output_layer(&mut self, l_chunk: (Vec<Vec<Weight>>, Vec<f64>)) {
        let weights = l_chunk.0;
        let biases = l_chunk.1;
        let mut nodes: Vec<Node> = vec![];
        let mut in_node_w_holder = vec![];

        for i in 0..weights[0].len() {
            let mut in_bank = vec![];
            for w in weights.iter() {
                in_bank.push(w[i]);
            }
            in_node_w_holder.push(in_bank);
        }

        for i in 0..weights[0].len() {
            nodes.push((
                (0f64, ActivationType::Sigmoid),
                vec![],
                in_node_w_holder.clone()[i].clone(),
                0,
                weights[0].len(),
                biases[i]
            ));
        }

        self.0.push((nodes, weights[0].len()));
        //checked and weights connected correctly according to test in main
    }

    //fn forward_pass() {
    //    todo!()
    //}
}



fn build_network_from_txt_file(txt_file_name: &str) -> Network {
    let file = fs::read_to_string(txt_file_name);
    //add_layer takes two l_chunks
    //an l_chunk represents the weights from the prev layers nodes, the first ind being the node in the prev layer
    //  and the second ind being the node in this layer/
    //in addition an l_chunk has the bias of the layer, the first of the two passed will be used for
    //  the layer being added to the network

    let mut layer_sizes: Vec<usize> = vec![];
    let mut w_block_holder: Vec<Vec<Vec<Weight>>> = vec![];
    let mut b_block_holder: Vec<Vec<f64>> = vec![];
    let mut w_block_builder: Vec<Vec<Weight>> = vec![];
    let mut b_block_builder: Vec<f64> = vec![];

    for l in file.unwrap().lines() {
        if l.starts_with("#") {
            continue
        }
        if l.starts_with("[l_sizes]") {
            let sizes = l.strip_prefix("[l_sizes]").unwrap();
            let sizes = sizes.strip_suffix("[\\l_sizes]").unwrap();
            let sizes = sizes.split(",").collect::<Vec<&str>>();
            for e in sizes.iter() {
                layer_sizes.push(usize::from_str(e).unwrap());
            }
            continue
        }
        if l.starts_with("[w]") {
            w_block_builder = vec![];
            b_block_builder = vec![];
            let weight_chunks = l.strip_prefix("[w]").unwrap();
            let weight_chunks = weight_chunks.strip_suffix("[\\w]").unwrap();
            let weight_chunks = weight_chunks.split("|").collect::<Vec<&str>>();
            for e in weight_chunks {
                let inner_split = e.split(",").collect::<Vec<&str>>();
                let mut w_block_loop: Vec<Weight> = vec![];
                for w in inner_split {
                    w_block_loop.push(Option::Some(w.parse::<f64>().unwrap()));
                }
                w_block_builder.push(w_block_loop);
            }
            w_block_holder.push(w_block_builder);
        }
        if l.starts_with("[b]") {
            b_block_builder = vec![];
            let biases = l.strip_prefix("[b]").unwrap();
            let biases = biases.strip_suffix("[\\b]").unwrap();
            let biases = biases.split(",").collect::<Vec<&str>>();
            for e in biases {
                b_block_builder.push(e.parse::<f64>().unwrap());
            }
            b_block_holder.push(b_block_builder);
        }
    }

    let mut l_blocks = vec![];
    for i in 0..layer_sizes.len()-1 {
        l_blocks.push((w_block_holder[i].clone(),b_block_holder[i].clone()));
    }

    let layer_sizes: Vec<usize> = layer_sizes; //not mut anymore

    let mut layers = Network::start_network(layer_sizes[0]);
    for i in 0..layer_sizes.len()-1 {
        if i == 0 {
            for node_ind in 0..layers.0[0].0.len() {
                layers.0[0].0[node_ind].1 = l_blocks[i].0[node_ind].clone();
                layers.0[0].0[node_ind].3 = l_blocks[i].0[node_ind].len();
            }
        }
        if i == layer_sizes.len()-2 {
            layers.add_output_layer(l_blocks[i].clone());
        }
        else {
            layers.add_hidden_layer((l_blocks[i].clone(), l_blocks[i+1].clone()));
        }
    }
    layers
}


fn main() {
    let file: &str = "src/test_network.txt";
    let network: Network = build_network_from_txt_file(file);

    println!("{:?},{:?}",network.0.len(), network.0[0].0[0].1.len());
    let mut l_count = 0;
    let mut n_count = 0;
    for l in network.0.clone() {
        for n in l.0.clone() {
            println!("{:?},{:?}",&l_count, &n_count);
            dbg!(n.get_nv_pairs());
            n_count += 1;
        }
        l_count += 1;
        n_count = 0;
    }
}








