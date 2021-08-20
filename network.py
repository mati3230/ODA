import graph_nets as gn
import sonnet as snt
import tensorflow as tf
import numpy as np

from base_network import BaseNetwork


def attention2(input_graphs, model_fn, training):
    nodes_at_sender_edges = gn.blocks.broadcast_sender_nodes_to_edges(input_graphs)
    nodes_at_receiver_edges = gn.blocks.broadcast_receiver_nodes_to_edges(input_graphs)
    
    V = model_fn(nodes_at_sender_edges, is_training=training)
    U = model_fn(nodes_at_receiver_edges, is_training=training)
    U_a, _, _ = fast_dot(U, V)
    nominator = U_a

    temporary_graph_sent = input_graphs.replace(edges=nominator)
    denominator = gn.blocks.ReceivedEdgesToNodesAggregator(tf.math.unsorted_segment_sum)(temporary_graph_sent)
    denominator = 1 / (denominator + 1e-6)
    deno_graphs = input_graphs.replace(nodes=denominator)
    deno_edges = gn.blocks.broadcast_receiver_nodes_to_edges(deno_graphs)
    att = nominator * deno_edges
    return att


def graph_convolution2(model_fn_node, model_fn_neigh, activation, input_graphs, training, att_model_fn=None):
    # Send the node features to the edges that are being sent by that node. 
    nodes_at_sender_edges = gn.blocks.broadcast_sender_nodes_to_edges(input_graphs)
    if att_model_fn is not None:
        att = attention2(input_graphs=input_graphs, model_fn=att_model_fn, training=training)
        nodes_at_sender_edges *= att[:, None]

    temporary_graph_sent = input_graphs.replace(edges=nodes_at_sender_edges)

    # Average the all of the edges received by every node.
    nodes_with_aggregated_edges = gn.blocks.ReceivedEdgesToNodesAggregator(tf.math.unsorted_segment_sum)(temporary_graph_sent)

    z_neigh = model_fn_neigh(nodes_with_aggregated_edges, is_training=training)
    z_node = model_fn_node(input_graphs.nodes, is_training=training)
    updated_nodes = z_node + z_neigh
    if activation is not None:
        updated_nodes = activation(updated_nodes)


    output_graphs = input_graphs.replace(nodes=updated_nodes)

    return output_graphs


def fast_dot(a, b):
    dot = a*b
    dot = tf.reduce_sum(dot, axis=-1)
    a_n = tf.linalg.norm(a, axis=-1)
    b_n = tf.linalg.norm(b, axis=-1)
    dot /= ((a_n * b_n) + 1e-6)
    return dot, a_n, b_n


class FFGraphNet(BaseNetwork):
    def __init__(
            self,
            name,
            n_ft_outpt,
            n_actions,
            seed=None,
            trainable=True,
            check_numerics=False,
            initializer="glorot_uniform",
            mode="full",
            stateful=False,
            discrete=True,
            head_only=False,
            observation_size=None):
        super().__init__(
            name=name,
            n_ft_outpt=n_ft_outpt,
            n_actions=n_actions,
            seed=seed,
            trainable=trainable,
            check_numerics=check_numerics,
            initializer=initializer,
            stateful=stateful,
            mode=mode,
            discrete=discrete,
            head_only=head_only,
            observation_size=observation_size)

    def action(self, obs, training=False, decision_boundary=0.5):
        if training:
            noise = tf.random.normal(shape=obs.nodes.shape, stddev=0.1)
            in_nodes = obs.nodes + noise
            obs.replace(nodes=in_nodes)
        out_g1 = graph_convolution2(
            model_fn_node=self.model_fn_node_1,
            model_fn_neigh=self.model_fn_neigh_1,
            activation=tf.nn.relu,
            input_graphs=obs,
            training=training,
            att_model_fn=self.model_fn_att_1)
        out_g2 = graph_convolution2(
            model_fn_node=self.model_fn_node_2,
            model_fn_neigh=self.model_fn_neigh_2,
            activation=None,
            input_graphs=out_g1,
            training=training,
            att_model_fn=self.model_fn_att_2)
        out_g2_n = tf.concat([out_g2.nodes, out_g1.nodes], axis=-1)

        out_g2.replace(nodes=out_g2_n)

        half_e = int(obs.n_edge[0] / 2)
        s = obs.senders[:half_e]
        r = obs.receivers[:half_e]

        fi = tf.gather(out_g2.nodes, indices=s)
        fj = tf.gather(out_g2.nodes, indices=r)
        dot, fin, fjn = fast_dot(fi, fj)
        
        d = (dot + 1) / 2
        action = tf.where(d > decision_boundary, 1, 0)
        pi_action = {
            "action": action,
            "probs": d,
            "dot": dot
            }
        return pi_action

    def init_variables(
            self,
            name,
            n_ft_outpt,
            n_actions,
            trainable=True,
            seed=None,
            initializer="glorot_uniform",
            mode="full"):
        drop = 0.1
        self.model_fn_node_1 = snt.nets.MLP(
            output_sizes=[512, 256, 128],
            activation=tf.nn.relu,
            w_init=snt.initializers.TruncatedNormal(mean=0, stddev=1, seed=seed),
            dropout_rate=drop,
            activate_final=True,
            name="mlp_node_1"
            )
        self.model_fn_node_2 = snt.nets.MLP(
            output_sizes=[64, 16],
            activation=tf.nn.relu,
            w_init=snt.initializers.TruncatedNormal(mean=0, stddev=1, seed=seed),
            dropout_rate=drop,
            activate_final=False,
            name="mlp_node_2"
            )
        self.model_fn_neigh_1 = snt.nets.MLP(
            output_sizes=[512, 256, 128],
            activation=tf.nn.relu,
            w_init=snt.initializers.TruncatedNormal(mean=0, stddev=1, seed=seed),
            dropout_rate=drop,
            activate_final=True,
            name="mlp_neigh_1"
            )
        self.model_fn_neigh_2 = snt.nets.MLP(
            output_sizes=[64, 16],
            activation=tf.nn.relu,
            w_init=snt.initializers.TruncatedNormal(mean=0, stddev=1, seed=seed),
            dropout_rate=drop,
            activate_final=False,
            name="mlp_neigh_2"
            )
        self.model_fn_att_1 = snt.nets.MLP(
            output_sizes=[128],
            activation=tf.nn.relu,
            #w_init=snt.initializers.TruncatedNormal(mean=0, stddev=0.2, seed=seed),
            dropout_rate=drop,
            activate_final=False,
            name="mlp_att_1",
            with_bias=False
            )
        self.model_fn_att_2 = snt.nets.MLP(
            output_sizes=[64],
            activation=tf.nn.relu,
            #w_init=snt.initializers.TruncatedNormal(mean=0, stddev=0.2, seed=seed),
            dropout_rate=drop,
            activate_final=False,
            name="mlp_att_2",
            with_bias=False
            )

    def init_net(
            self,
            name,
            n_ft_outpt,
            seed=None,
            trainable=True,
            check_numerics=False,
            initializer="glorot_uniform",
            mode="full"):
        pass

    def get_vars(self, with_non_trainable=True):
        vars_ = []
        vars_.extend(self.model_fn_node_1.variables)
        vars_.extend(self.model_fn_neigh_1.variables)
        vars_.extend(self.model_fn_node_2.variables)
        vars_.extend(self.model_fn_neigh_2.variables)
        vars_.extend(self.model_fn_att_1.variables)
        vars_.extend(self.model_fn_att_2.variables)
        return vars_

    def reset(self):
        pass

    def preprocess(self, obs):
        pass

    def snapshot(self, obs, directory, filename):
        pass