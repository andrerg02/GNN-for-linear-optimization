import torch
import numpy as np
import torch.nn as nn

from torch_geometric.nn import MLP
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn.conv import MessagePassing


class BipartiteData(Data):
    '''
    A class for weighted batched bipartite graphs.

    Attributes:
        x_s : torch.tensor
            features of the left nodes

        x_t : torch.tensor
            features of the right nodes

        edge_index : torch.tensor
            edge indexes

        edge_weight : torch.tensor
            edge weights
    '''

    def __init__(self, x_s, x_t, edge_index, edge_weight, num_nodes):
        '''import sklearn
        Parameters

        x_s : torch.tensor
            features of the left nodes

        x_t : torch.tensor
            features of the right nodes

        edge_index : torch.tensor
            edge indexes

        edge_weight : torch.tensor
            edge weights

        '''
        super().__init__(num_nodes=num_nodes)
        self.x_s = x_s
        self.x_t = x_t
        self.edge_index = edge_index
        self.edge_weight = edge_weight

    def __inc__(self, key, value, *args, **kwargs):
        '''
        Increments the size of the graph.

        Parameters:
            key : str
                key of the attribute to be incremented

            value : torch.tensor
                value of the attribute to be incremented

        Returns:
            The incremented size of the graph.
        '''
        if key == 'edge_index':
            return torch.tensor([[self.x_s.size(0)], [self.x_t.size(0)]])
        return super().__inc__(key, value, *args, **kwargs)


class GCNConv(MessagePassing):
    '''
    A GNN Convolutional Layer for weighted batched bipartite graphs.
    ------------------------------------------------------------------------------------------------------------------

    Attributes:
        in_channels : int
            number of input features

        hidden_channels : int
            number of hidden features

        out_channels : int
            number of output features

    ------------------------------------------------------------------------------------------------------------------
    '''

    def __init__(self, in_channels, hidden_channels, out_channels):
        '''
        Parameters
        in_channels : int
            number of input features

        hidden_channels : int
            number of hidden features

        out_channels : int
            number of output features
        '''
        super(GCNConv, self).__init__(aggr=None)
        self.mlp1 = MLP([in_channels + out_channels,
                        hidden_channels, out_channels])
        self.mlp2 = MLP([in_channels, hidden_channels, out_channels])

    def forward(self, x, edge_index, edge_weight, size):
        '''
        Forward pass of the layer.

        Parameters:
            x : torch.tensor
                tuple with the features

            edge_index : torch.tensor
                edge indexes

            edge_weight : torch.tensor
                edge weights

            size : tuple
                size of the graph

        Returns:
            The updated features of x[0].
        '''
        return self.propagate(edge_index, x=x, edge_weight=edge_weight, size=size)

    def message(self, x, edge_weight):
        '''
        Message function of the layer.

        Parameters:
            x : torch.tensor
                tuple with the features. Messages are passed from x[0] to x[1]

            edge_weight : torch.tensor
                edge weights

        Returns:
            The messages passed from x[1] to x[0].
        '''
        reshaped = edge_weight[:, :, None].reshape((edge_weight.shape[0] * edge_weight.shape[1]) // self.mlp2(x[1]).shape[0],
                                                   self.mlp2(x[1]).shape[0],
                                                   1)
        a = torch.mul(reshaped, self.mlp2(x[1]))
        a = a.reshape(x[0].shape[0], (a.shape[0] * a.shape[1]) //
                      x[0].shape[0], a.shape[2])
        return a

    def aggregate(self, inputs):
        '''
        Aggregation function of the layer.

        Parameters:
            inputs : torch.tensor
                messages passed from x[1] to x[0]

        Returns:
            The aggregated messages.
        '''
        return torch.sum(inputs, dim=1)

    def update(self, aggr_out, x):
        '''
        Update function of the layer.

        Parameters:
            aggr_out : torch.tensor
                aggregated messages

            x : torch.tensor
                tuple with the features

        Returns:
            The updated features of x[0].
        '''
        return self.mlp1(torch.cat((x[0], aggr_out), dim=1))


class LPGCN(nn.Module):
    ''' 
    A GNN model for solving linear programs. The LP is modeled as a weighted bipartite graph with constraints on the left and
    variables on the right. The optimization problems are of the form: min c^T x s.t. Ax (constraints) b, l <= x <= u
    ------------------------------------------------------------------------------------------------------------------

    Attributes:
        num_constraints : int
            number of constraints in the LP

        num_variables : int
            number of variables in the LP

        num_layers : int
            number of layers in the GNN
    ------------------------------------------------------------------------------------------------------------------
    '''

    def __init__(self, num_constraints, num_variables, num_layers=5):
        '''
        Parameters
        num_constraints : int
            number of constraints in the LP

        num_variables : int
            number of variables in the LP

        num_layers : int
            number of layers in the GNN
        '''

        super().__init__()

        self.num_constraints = num_constraints
        self.num_variables = num_variables

        self.num_layers = num_layers

        # Generate random integers for the dimensions of the hidden layers.
        # The dimensions are powers of 2, with min = 2 and max = 512.
        ints = np.random.randint(1, 10, size=self.num_layers)
        dims = [2 ** i for i in ints]

        # Encode the input features into the embedding space
        self.fv_in = MLP([2, 32, dims[0]])
        self.fw_in = MLP([3, 32, dims[0]])

        # Hidden states (convolutions)
        self.cv = nn.ModuleList([GCNConv(dims[l-1], 32, dims[l])
                                for l in range(1, self.num_layers)])
        self.cw = nn.ModuleList([GCNConv(dims[l-1], 32, dims[l])
                                for l in range(1, self.num_layers)])

        # Feas and obj output function
        self.f_out = MLP([2 * dims[self.num_layers-1], 32, 1])

        # Sol output function
        self.fw_out = MLP([3 * dims[self.num_layers-1], 32, 1])

    def construct_graph(self, c, A, b, constraints, l, u):
        '''
        Constructs the bipartite graph of the LP.

        Parameters:
            c : torch.tensor
                objective function coefficients

            A : torch.tensor
                constraint matrix

            b : torch.tensor
                right-hand side values for the constraints

            constraints : torch.tensor
                constraint types (0 for <= and 1 for =)

            l : torch.tensor
                lower bounds for the variables

            u : torch.tensor
                upper bounds for the variables
        '''

        # Constraint features
        hv = torch.cat((b.unsqueeze(2), constraints.unsqueeze(2)), dim=2)

        # Variable features
        hw = torch.cat((c.unsqueeze(2), l.unsqueeze(2), u.unsqueeze(2)), dim=2)

        # Edges
        E = A

        return hv, hw, E

    def init_features(self, hv, hw):
        '''
        Initializes the features of the nodes (layer 0).

        Parameters:
            hv : torch.tensor
                constraint features

            hw : torch.tensor
                variable features

        Returns:
            The initialized features of the nodes.
        '''

        # Applies MLP to each line of constraint features
        hv_0 = []
        for i in range(self.num_constraints):
            hv_0.append(self.fv_in(hv[:, i]))

        # Applies MLP to each line of variable features
        hw_0 = []
        for j in range(self.num_variables):
            hw_0.append(self.fw_in(hw[:, j]))

        hv = torch.stack(hv_0, dim=1)
        hw = torch.stack(hw_0, dim=1)

        return hv, hw

    def convs(self, hv, hw, edge_index, E, layer, batch_size):
        '''
        Performs the left-to-right and right-to-left convolutions for each layer.

        Parameters:
            hv : torch.tensor
                constraint features

            hw : torch.tensor
                variable features

            edge_index : torch.tensor
                edge indexes

            E : torch.tensor
                edge weights

            layer : int
                layer index

            batch_size : int
                batch size

        Returns:
            The updated constraint and variable features.
        '''

        hv_l = self.cv[layer]((hv, hw),
                              edge_index,
                              E,
                              (self.num_constraints * batch_size, self.num_variables))

        hw_l = self.cw[layer]((hw, hv),
                              torch.flip(edge_index, dims=[1, 0]),
                              E.T,
                              (self.num_variables, self.num_constraints * batch_size))

        return hv_l, hw_l

    def single_output(self, hv, hw):
        '''
        Output for feasibility and objective functions.

        Parameters:
            hv : torch.tensor
                constraint features

            hw : torch.tensor
                variable features

        Returns:
            If the output is for feasibility, returns a binary vector indicating if the LP's are feasible or not.
            If the output is for objective, returns the objective function value of the LP's.
        '''

        y_out = self.f_out(
            torch.cat((torch.sum(hv, 1), torch.sum(hw, 1)), dim=1))

        return y_out

    def sol_output(self, hv, hw):
        '''
        Output for solution function.

        Parameters:
            hv : torch.tensor
                constraint features

            hw : torch.tensor
                variable features

        Returns:
            Returns the approximated solution of the LP's.
        '''

        sol = []
        for j in range(self.num_variables):
            joint = torch.cat(
                (torch.sum(hv, 1), torch.sum(hw, 1), hw[:, j]), dim=1)
            sol.append(self.fw_out(joint))

        sol = torch.stack(sol, dim=1)

        return sol[:, :, 0]

    def forward(self, c, A, b, constraints, l, u, edge_index, phi='feas'):
        '''
        Forward pass of the model.

        Parameters:
            c : torch.tensor
                objective function coefficients

            A : torch.tensor
                constraint matrix

            b : torch.tensor
                right-hand side values for the constraints

            constraints : torch.tensor
                constraint types (0 for <= and 1 for =)

            l : torch.tensor
                lower bounds for the variables

            u : torch.tensor
                upper bounds for the variables

            edge_index : torch.tensor
                edge indexes

            phi : str
                type of function (feas, obj or sol)

        Returns:
            If the output is for feasibility, returns a binary vector indicating if the LP's are feasible or not.
            If the output is for objective, returns the objective function value of the LP's.
            If the output is for solution, returns the approximated solution of the LP's.
        '''

        hv, hw, E = self.construct_graph(c, A, b, constraints, l, u)
        hv, hw = self.init_features(hv, hw)

        batch_size = hv.shape[0]

        graphs = [BipartiteData(x_s=hv[i], x_t=hw[i], edge_index=edge_index[i].T, edge_weight=E[i],
                                num_nodes=self.num_variables+self.num_constraints) for i in range(hv.shape[0])]
        loader = DataLoader(graphs, batch_size=batch_size)
        batch = next(iter(loader))

        hv = batch.x_s
        hw = batch.x_t
        edge_index = batch.edge_index
        E = batch.edge_weight

        # Iterates over the layers
        for l in range(self.num_layers-1):
            hv, hw = self.convs(hv, hw, edge_index, E, l, batch_size)

        hv = hv.reshape(batch_size, hv.shape[0] // batch_size, hv.shape[1])
        hw = hw.reshape(batch_size, hw.shape[0] // batch_size, hw.shape[1])

        if phi == 'feas':
            output = self.single_output(hv, hw)
            bins = [1 if elem >= 1/2 else 0 for elem in output]
            return torch.tensor(bins, dtype=torch.float32, requires_grad=True)

        elif phi == 'obj':
            return self.single_output(hv, hw)

        elif phi == 'sol':
            return self.sol_output(hv, hw)

        else:
            return "Please, choose one type of function: feas, obj or sol"
