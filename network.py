import graph_nets as gn
import sonnet as snt
import tensorflow as tf
import numpy as np

from base_network import BaseNetwork


def graph_convolution(model_fn, input_graphs, training, node_factor=0.7):
    # Send the node features to the edges that are being sent by that node. 
    nodes_at_sender_edges = gn.blocks.broadcast_sender_nodes_to_edges(input_graphs)
    temporary_graph_sent = input_graphs.replace(edges=nodes_at_sender_edges)

    nodes_at_receiver_edges = gn.blocks.broadcast_receiver_nodes_to_edges(input_graphs)
    temporary_graph_recv = input_graphs.replace(edges=nodes_at_receiver_edges)

    # Average the all of the edges received by every node.
    nodes_with_aggregated_edges_s = gn.blocks.ReceivedEdgesToNodesAggregator(tf.math.unsorted_segment_mean)(temporary_graph_sent)
    # Average the all of the edges sent by every node.
    nodes_with_aggregated_edges_r = gn.blocks.SentEdgesToNodesAggregator(tf.math.unsorted_segment_mean)(temporary_graph_recv)
    nodes_with_aggregated_edges = nodes_with_aggregated_edges_s + nodes_with_aggregated_edges_r

    # Interpolation between input and neighbour features
    aggregated_nodes = node_factor * input_graphs.nodes + (1 - node_factor) * nodes_with_aggregated_edges
    updated_nodes = model_fn(aggregated_nodes, is_training=training)

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
        out_g = graph_convolution(model_fn=self.model_fn, input_graphs=obs, training=training, node_factor=0.7)
        s = out_g.senders
        r = out_g.receivers

        fi = tf.gather(out_g.nodes, indices=s)
        fj= tf.gather(out_g.nodes, indices=r)
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
        self.model_fn = snt.nets.MLP(
            output_sizes=[512, 256, 128, 64, 16],
            activation=tf.nn.relu,
            w_init=snt.initializers.TruncatedNormal(mean=0, stddev=0.2, seed=seed),
            dropout_rate=0.2,
            activate_final=False,
            name="mlp1"
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
        vars_.extend(self.model_fn.variables)
        return vars_

    def reset(self):
        pass

    def preprocess(self, obs):
        pass

    def snapshot(self, obs, directory, filename):
        pass