from .base import Segmenter

import numpy as np
import nifty
import nifty.graph.opt.multicut as nmc
import nifty.graph.opt.lifted_multicut as nlmc
from nifty import Configuration


# TODO different exponent for edge weighting ?
def transform_probabilities_to_costs(probabilities,
                                     beta=.5,
                                     edge_sizes=None,
                                     weighting_exponent=1.):
    p_min = 0.001
    p_max = 1. - p_min
    costs = (p_max - p_min) * probabilities + p_min
    # probabilities to costs, second term is boundary bias
    costs = np.log((1. - costs) / costs) + np.log((1. - beta) / beta)
    # weight the costs with edge sizes, if they are given
    if edge_sizes is not None:
        assert len(edge_sizes) == len(costs)
        w = edge_sizes / edge_sizes.max()
        if weighting_exponent != 1.:
            w = w**weighting_exponent
        costs *= w
    return costs


class Multicut(Segmenter):
    solvers = ["greedy-additive", "kernighan-lin", "fusion-moves", "ilp"]

    def __init__(self, solver, beta=.5, weight_edges=True, **solver_options):
        assert solver in self.solvers
        if solver == "ilp":
            assert self._have_ilp()
        self.solver = solver
        assert 0. < beta < 1.
        self.beta = beta
        self.weight_edges = weight_edges
        self.solver_options = solver_options

    def probabilities_to_costs(self, probabilities, edge_sizes=None):
        if self.weight_edges:
            assert edge_sizes is not None
            return transform_probabilities_to_costs(probabilities, self.beta, edge_sizes)
        else:
            return transform_probabilities_to_costs(probabilities, self.beta)

    @staticmethod
    def _have_ilp():
        return Configuration.WITH_CPLEX or Configuration.WITH_GUROBI or Configuration.WITH_GLPK

    def _get_fusion_moves(self, objective):
        # get the proposal generator
        gen = self.solver_options.get("proposal_generator", None)
        if gen == "watershed" or gen is None:
            sigma = self.solver_options.get("sigma", 1.)
            n_seeds = self.solver_options.get("n_seeds", 1.)
            generator = objective.watershedCcProposals(sigma, n_seeds)
        elif gen == "interface_flipper":
            generator = objective.interFaceFlipperCcProposals()
        elif gen == "random_node_color":
            n_colors = self.solver_options.get("n_colors", 2)
            generator = objective.randomNodeColorProposals(n_colors)
        else:
            raise RuntimeError("Invalid fusion move proposal generator")

        # get the solver backend
        backend = self.solver_options.get("backend", "kernighan-lin")

        if backend == "greedy-additive":
            solver_backend = objective.greedyAdditiveFactory()
        elif backend == "kernighan-lin":
            solver_backend = objective.kernighanLinFactory(warmstartGreedy=True)
        elif backend == "ilp":
            assert self._have_ilp()
            solver_backend = objective.multicutIlpFactory()
        else:
            raise RuntimeError("Invalid fusion move backend")

        # get the other options
        n_threads = self.solver_options.get("n_threads", 1)
        n_iter = self.solver_options.get("n_iter", 100)
        n_stop = self.solver_options.get("n_stop", 10)

        fm = objective.ccFusionMoveBasedFactory(proposalGenerator=generator,
                                                numberOfThreads=n_threads,
                                                numberOfIterations=n_iter,
                                                stopIfNoImprovement=n_stop,
                                                fusionMove=solver_backend)
        # TODO need to change ws verbosity if we use verbosity
        # chain solvers for warmstarting
        warmstart_greedy = self.solver_options.get("warmstart_greedy", True)
        warmstart_kl = self.solver_options.get("warmstart_kl", True)
        if warmstart_greedy and warmstart_kl:
            ga_factory = objective.greedyAdditiveFactory()
            kl_factory = objective.kernighanLinFactory()
            fm = objective.chainedSolversFactory([ga_factory, kl_factory, fm])
        elif warmstart_greedy:
            ga_factory = objective.greedyAdditiveFactory()
            fm = objective.chainedSolversFactory([ga_factory, fm])
        elif warmstart_kl:
            kl_factory = objective.kernighanLinFactory()
            fm = objective.chainedSolversFactory([kl_factory, fm])

        return fm.create(objective)

    # TODO kwarg for verbosity
    # TODO support logging visitor
    def _segmentation_impl(self, graph, costs, time_limit=None, **kwargs):
        objective = nmc.multicutObjective(graph, costs)

        if self.solver == 'greedy-additive':
            solver_impl = objective.greedyAdditiveFactory().create(objective)
        elif self.solver == 'kernighan-lin':
            warmstart = self.solver_options.get('warmstart_greedy', True)
            # TODO need to set this if we use verbosity
            # greedyVisitNth = kwargs.pop('greedyVisitNth', 100)
            solver_impl = objective.kernighanLinFactory(warmStartGreedy=warmstart).create(objective)
        elif self.solver == 'fusion-moves':
            solver_impl = self._get_fusion_moves(objective)
        elif self.solver == 'ilp':
            ilp_backend = self.solver_options.get("ilp_backend", None)
            solver_impl = objective.multicutIlpFactory(ilpSolver=ilp_backend).create(objective)

        # TODO this needs to change once we suport verbosity / logging
        if time_limit is None:
            node_labels = solver_impl.optimize()
        else:
            visitor = objective.verboseVisitor(visitNth=100000000, timeLimitSolver=time_limit)
            node_labels = solver_impl.optimize(visitor=visitor)

        return node_labels


# TODO this does not fit into th `Segmenter` scheme that easily
# for now, it doesn't inherit from it
class LiftedMulticut(object):
    solvers = ["greedy-additive", "kernighan-lin", "fusion-moves"]

    def __init__(self, solver, beta=.5, weight_edges=True, **solver_options):
        assert solver in self.solvers
        self.solver = solver
        assert 0. < beta < 1.
        self.beta = beta
        self.weight_edges = weight_edges
        self.solver_options = solver_options

    def probabilities_to_costs(self, probabilities, edge_sizes=None):
        if self.weight_edges:
            assert edge_sizes is not None
            return transform_probabilities_to_costs(probabilities, self.beta, edge_sizes)
        else:
            return transform_probabilities_to_costs(probabilities, self.beta)

    def _get_fusion_moves(self, objective):
        seeding_strategy = self.solver_options.get('seeding-strategy', 'SEED_FROM_LOCAL')
        sigma = self.solver_options.get('sigma', 10.)
        seed_fraction = self.solver_options.get('seed_fraction', .1)
        pgen = objective.watershedProposalGenerator(seedingStrategy=seeding_strategy,
                                                    sigma=sigma,
                                                    numberOfSeeds=seed_fraction)
        # we leave the number of iterations at default values for now
        return objective.fusionMoveBasedFactory(proposalGenerator=pgen).create(objective)

    # TODO kwarg for verbosity
    # TODO support logging visitor
    def __call__(self, local_uvs, lifted_uvs,
                 local_costs, lifted_costs,
                 time_limit=None, **kwargs):
        n_nodes = int(local_uvs.max()) + 1
        graph = nifty.graph.undirectedGraph(n_nodes)
        graph.insertEdges(local_uvs)
        objective = nlmc.liftedMulticutObjective(graph)

        # TODO local vs. lifted weighting ?!

        objective.setCosts(local_uvs, local_costs)
        objective.setCosts(lifted_uvs, lifted_costs)

        # TODO visitors and time limit!
        # TODO would be nice to have solver chaining in nifty for lmc too
        solver_ga = objective.greedyAdditiveFactory().create(objective)
        # first solve greedy-agglomerative
        node_labels = solver_ga.optimize()
        if self.solver == 'greedy-additive':
            return node_labels
        else:
            solver_kl = objective.liftedMulticutKernighanLinFactory().create(objective)
            node_labels = solver_kl.optimize(node_labels)
            if self.solver == 'kernighan-lin':
                return node_labels
            elif self.solver == 'fusion-moves':
                solver_fm = self._get_fusion_moves(objective)
                node_labels = solver_fm.optimize(node_labels)

        return node_labels
