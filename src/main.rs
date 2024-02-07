use std::f64::consts;
use std::fs;
use std::str::FromStr;
use rand::random;

#[derive(Copy, Clone, PartialEq, Debug)]
enum ActivationType {
    Sigmoid,
    TanH,
    Linear,
    Squared,
    Cubed,
}

type PreActivationValue = (f64,ActivationType);

trait PAVUtils {
    fn activate_pav(&self) -> Option<(f64, f64)>;
}
impl PAVUtils for PreActivationValue {
    fn activate_pav(&self) -> Option<(f64, f64)> {
        match self.1.clone() {
            ActivationType::Sigmoid => {
                let post_activation = 1.0 / (1.0 + consts::E.powf(-self.0));
                let post_prime_activation = consts::E.powf(self.0)
                    / (1.0 + consts::E.powf(self.0)).powf(2.0);
                Option::Some((post_activation,post_prime_activation))
            }
            ActivationType::TanH => {
                let post_activation = self.0.tanh();
                let post_prime_activation = 1f64 - self.0.tanh().powi(2);
                Option::Some((post_activation,post_prime_activation))
            }
            ActivationType::Linear => {
                Option::Some((self.0,1.0))
            }
            ActivationType::Squared => {
                let post_activation = self.0.powi(2);
                let post_prime_activation = 2f64 * self.0;
                Option::Some((post_activation,post_prime_activation))
            }
            ActivationType::Cubed => {
                let post_activation = self.0.powi(3);
                let post_prime_activation = 3f64 * self.0.powi(2);
                Option::Some((post_activation,post_prime_activation))
            }
        }
    }
}

type Node = (PreActivationValue, Vec<Weight>, Vec<Weight>, usize, usize, f64, Option<f64>, Option<f64>, Option<f64>);
//0 - preactval, 1 - outgoing weights, 2 - incoming weights, 3 - next layer size, 4 - prev layer size,
//  5 - bias, 6 - biasDelta, 7 - errorDelta (during bp), 8 - post prime activation value
trait NodeUtils {
    fn activate_node(&self) -> (f64, f64);
    fn set_pav(&mut self, v: f64, t: ActivationType);
    fn get_pav(&self) -> PreActivationValue;
    fn get_outgoing_weights(&self) -> Vec<Weight>;
    fn set_outgoing_weights(&mut self, weights: Vec<Weight>);
    fn get_incoming_weights(&self) -> Vec<Weight>;
    fn set_incoming_weights(&mut self, weights: Vec<Weight>);
    fn get_size_next_layer(&self) -> usize;
    fn set_size_next_layer(&mut self, size: usize);
    fn get_size_prev_layer(&self) -> usize;
    fn get_bias(&self) -> f64;
    fn get_bias_delta(&self) -> Option<f64>;
    fn get_error_delta(&self) -> Option<f64>;
    fn set_error_delta(&mut self, val: f64);
    fn set_bias_delta(&mut self, val: f64);
    fn forward(&mut self) -> Vec<f64>;
    fn set_pp_av(&mut self, v: f64);
    fn get_pp_av(&self) -> Option<f64>;
}
impl NodeUtils for Node {
    fn activate_node(&self) -> (f64, f64) {
        let post_act_val = &self.get_pav().activate_pav();
        match post_act_val {
            None => {panic!()},
            Some(x) => {return *x;},
        }
    }
    fn set_pav(&mut self, v: f64, t: ActivationType) {
        self.0.0 = v;
        self.0.1 = t;
    }
    fn get_pav(&self) -> PreActivationValue {
        self.0
    }
    fn get_outgoing_weights(&self) -> Vec<Weight> {
        self.1.clone()
    }
    fn set_outgoing_weights(&mut self, weights: Vec<Weight>) {
        self.1 = weights
    }
    fn get_incoming_weights(&self) -> Vec<Weight> {
        self.2.clone()
    }
    fn set_incoming_weights(&mut self, weights: Vec<Weight>) {
        self.2 = weights
    }
    fn get_size_next_layer(&self) -> usize {
        self.3
    }
    fn set_size_next_layer(&mut self, size: usize){
        self.3 = size
    }
    fn get_size_prev_layer(&self) -> usize {
        self.4
    }
    fn get_bias(&self) -> f64 {
        self.5
    }
    fn get_bias_delta(&self) -> Option<f64> {
        return self.6;
    }
    fn get_error_delta(&self) -> Option<f64> {
        self.7
    }
    fn set_error_delta(&mut self, val: f64) {
        self.7 = Option::Some(val);
    }
    fn set_bias_delta(&mut self, val: f64) {
        self.6 = Option::Some(val);
    }

    fn forward(&mut self) -> Vec<f64> {
        //make empty list to eventually output
        let mut add_to_next_layer_pav: Vec<f64> = vec![];


        //get nv pairs (ind_of_node_to, (post_act, post_prime_act)) and weights
        let pair: (f64, f64) = self.activate_node();
        let weights = self.get_outgoing_weights();

        //for first one, update node to have post_act and post prime act
        for i in 0..weights.len() {
            match weights[i] {
                None => {
                    println!("{:?}","NONE WEIGHT VALUE REFERENCED");
                    return vec![];
                }
                Some(x) => {
                    let v = pair.0 * x;
                    add_to_next_layer_pav.push(v);
                    self.set_pp_av(pair.1);
                }
            }
        }
        add_to_next_layer_pav
    }

    fn set_pp_av(&mut self, v: f64) {
        self.8 = Option::Some(v);
    }

    fn get_pp_av(&self) -> Option<f64> {
        self.8
    }
}

type Layer = (Vec<Node>, usize);
trait LayerUtils {
    fn make(v: Vec<Node>) -> Layer;
    fn get_layer_size(&self) -> usize;
    fn get_nodes(&self) -> Vec<Node>;
}
impl LayerUtils for Layer {
    fn make(v: Vec<Node>) -> Layer {
        let l = *&v.len().clone();
        (v, l)
    }
    fn get_layer_size(&self) -> usize {
        self.1
    }
    fn get_nodes(&self) -> Vec<Node> {
        self.0.clone()
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

type Network = (Vec<Layer>,Vec<usize>,Vec<ActivationType>);

trait NetworkUtils {
    fn start_network(input_layer_size: usize, act_types: Vec<ActivationType>) -> Network;
    fn add_hidden_layer(&mut self, l_chunk: ((Vec<Vec<Weight>>, Vec<f64>), (Vec<Vec<Weight>>, Vec<f64>)));
    fn add_output_layer(&mut self, l_chunk: (Vec<Vec<Weight>>, Vec<f64>));
    fn get_layers(&self) -> Vec<Layer>;
    fn forward_pass(&mut self) -> Vec<f64>;
    fn set_layer_values(&mut self, layer_ind: usize, values: Vec<f64>, activation_type: ActivationType);
    fn get_input_act_type(&self) -> ActivationType;
    fn get_hidden_act_type(&self) -> ActivationType;
    fn get_output_act_type(&self) -> ActivationType;
}
impl NetworkUtils for Network {
    fn start_network(input_layer_size: usize, act_types: Vec<ActivationType>) -> Network {
        let mut input_nodes: Vec<Node> = vec![];
        for _ in 0..input_layer_size {
            input_nodes.push(((0.0f64, ActivationType::Linear), vec![], vec![], 0, 0, 0f64, Option::None, Option::None, Option::None));
        }
        let input_layer: Layer = (input_nodes, input_layer_size);
        let network: Network = (vec![input_layer], vec![input_layer_size], act_types);
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
                biases[i],
                Option::None,
                Option::None,
                Option::None
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
                (0f64, ActivationType::TanH),
                vec![],
                in_node_w_holder.clone()[i].clone(),
                0,
                weights[0].len(),
                biases[i],
                Option::None,
                Option::None,
                Option::None
            ));
        }

        self.0.push((nodes, weights[0].len()));
        //checked and weights connected correctly according to test in main
    }
    fn get_layers(&self) -> Vec<Layer> {
        self.0.clone()
    }

    fn forward_pass(&mut self) -> Vec<f64> {
        for layer_ind in 0..(self.get_layers().len()-1) {
            let num_nodes_next_layer = self.get_layers()[layer_ind+1].get_nodes().len();
            let mut add_sums: Vec<f64> = vec![0.0f64; num_nodes_next_layer];

            for node_ind in 0..self.get_layers()[layer_ind].get_nodes().len() {
                //can't use get_layers or get_nodes here since the copy won't keep pp_av
                let node_outs = self.0[layer_ind].0[node_ind].forward();
                if node_ind == 0 {
                    add_sums = node_outs;
                }
                else {
                    for v in 0..node_outs.len() {
                        add_sums[v] += node_outs[v];
                    }
                }
            }

            for node_ind in 0..num_nodes_next_layer {
                add_sums[node_ind] += self.get_layers()[layer_ind+1].get_nodes()[node_ind].get_bias();
            }

            if layer_ind == self.get_layers().len()-2 {
                self.set_layer_values(layer_ind+1,add_sums.clone(),self.get_output_act_type());
                for i in 0..num_nodes_next_layer {
                    //set output layer post prime activation values to 1 (since they always are)
                    self.0[layer_ind+1].0[i].set_pp_av(1f64);
                }
                return add_sums.iter().map(|x| x.powi(2)).collect();
            }
            else {
                self.set_layer_values(layer_ind+1,add_sums,self.get_hidden_act_type());
            }
        }
        vec![]
    }

    fn set_layer_values(&mut self, layer_ind: usize, values: Vec<f64>, activation_type: ActivationType) {
        for i in 0..values.len() {
            //Can't use get_layers and get_nodes for SETTING values since they use clone!!!!!!!!!!!!!!!!!!
            //  self.get_layers()[0].get_nodes()[i].set_pav(values[i], ActivationType::Linear);
            //self.0 - layers
            //self.0[0] - input layer
            //self.0[0].0 - nodes
            self.0[layer_ind].0[i].set_pav(values[i], activation_type);
        }
    }

    fn get_input_act_type(&self) -> ActivationType {
        self.2[0]
    }
    fn get_hidden_act_type(&self) -> ActivationType {
        self.2[1]
    }
    fn get_output_act_type(&self) -> ActivationType {
        self.2[2]
    }
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

    // TEMP JUST ASSIGNING ACT TYPES HERE CHANGE SYNTAX TO HAVE IN TXT FILE AT SOME POINT!!!!!!!!!!!!!
    let mut layers = Network::start_network(layer_sizes[0], vec![ActivationType::Linear,ActivationType::Cubed,ActivationType::Squared]);
    for i in 0..layer_sizes.len()-1 {
        if i == 0 {
            for node_ind in 0..layers.0[0].0.len() {
                layers.0[0].0[node_ind].set_outgoing_weights(l_blocks[i].0[node_ind].clone());
                layers.0[0].0[node_ind].set_size_next_layer(l_blocks[i].0[node_ind].len());
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
    let file: &str = "src/test_network_xor.txt";
    let mut network: Network = build_network_from_txt_file(file);

    let example_input_values = vec![3.5f64,-1.5f64];
    network.set_layer_values(0, example_input_values, ActivationType::Linear);

    println!("{:?}",network.forward_pass());

    for l in network.get_layers() {
        for n in l.get_nodes() {
            println!("{:?}",n.8);
        }
    }
}








