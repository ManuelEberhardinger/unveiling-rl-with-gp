import numpy as np
from graphviz import Digraph, Source
import random
from copy import deepcopy
import parser
from scipy.special import softmax
from collections import defaultdict
from typing import List


class DepthExceededException(Exception):
    pass


class ProgramNotCorrectlyTypedException(Exception):
    def __init__(self, message: str):
        self.message = message
        super().__init__(message)


class ProgramNotCorrectlyRewrittenException(Exception):
    def __init__(self, message: str):
        self.message = message
        super().__init__(message)


class Node():
    def __init__(self, primitive, grammar):
        self.childern: List = []
        self.primitive = primitive
        self.grammar = grammar

    def __str__(self):
        return self.primitive.name

    def __repr__(self):
        return self.primitive.name

    def random_subtree(self, depth: int = 1, p=None, mutation=False, epsilon=0.5):
        if depth < 0:
            raise DepthExceededException()

        if mutation:
            # first change the primitive so that we have the possibiliy to also change terminals
            possible_childern = self.grammar.get_possible_childern(
                self.primitive.returns(), depth)
            probs = self.get_probabilities(
                possible_childern, p, epsilon=epsilon)
            prim = np.random.choice(possible_childern, 1, p=probs)[0]
            old_prim = self.primitive
            self.primitive = prim
            if prim.returns() == old_prim.returns() and prim.parameters() == old_prim.parameters():
                # only change childern if the types are different
                return

        self.childern = []
        for child_param in self.primitive.parameters():
            possible_childern = self.grammar.get_possible_childern(
                child_param, depth)
            probs = self.get_probabilities(
                possible_childern, p, epsilon=epsilon)
            prim = np.random.choice(possible_childern, 1, p=probs)[0]
            self.childern.append(Node(prim, self.grammar))
            self.childern[-1].random_subtree(depth - 1, p, mutation=False)

    def get_probabilities(self, childern, p, epsilon=0.5):
        if p is None or random.random() > epsilon:
            return None
        mean = np.mean([p.get(str(c))
                       for c in childern if p.get(str(c)) is not None])
        probs = [p.get(str(c), mean) for c in childern]
        soft = softmax(probs)
        return soft

    def size(self):
        if self.is_terminal() or self.is_input_parameter():
            return 1
        else:
            return 1 + sum([child.size() for child in self.childern])

    def distribution(self, dist):
        dist[str(self)] += 1
        for child in self.childern:
            child.distribution(dist)
        return dist

    def draw(self, dot, count):  # dot & count are lists in order to pass "by reference"
        node_name = str(count[0])
        dot[0].node(node_name, str(self))

        for child in self.childern:
            count[0] += 1
            dot[0].edge(node_name, str(count[0]))
            child.draw(dot, count)

    def mutate(self, p=None, prob=0.5, max_tries=10, is_root=False, epsilon=0.5):

        if random.random() < prob and not is_root:  # mutate at this node
            backup_primitive = deepcopy(self.primitive)
            backup_childern = deepcopy(self.childern)

            while max_tries > 0:
                try:
                    self.random_subtree(6, p, mutation=True, epsilon=epsilon)
                    break

                except DepthExceededException as e:
                    max_tries -= 1
                    self.primitive = backup_primitive
                    self.childern = backup_childern

        elif len(self.childern):
            random.choice(self.childern).mutate(p, prob, epsilon=epsilon)

    def merge(self, other, prob=0.1, p=None, depth=3, max_tries=10):
        if random.random() < prob:
            while max_tries > 0:
                try:
                    prim = np.random.choice(self.grammar.get_possible_childern(
                        'taction', depth, no_terminal=True), 1, p=p)[0]
                    node = Node(prim, self.grammar)
                    child_param = node.primitive.parameters()[0]
                    prim = np.random.choice(self.grammar.get_possible_childern(
                        child_param, depth), 1, p=p)[0]
                    node.childern = []
                    node.childern.append(Node(prim, self.grammar))
                    node.childern[-1].random_subtree(depth, p, mutation=False)
                    node.childern.append(self)
                    node.childern.append(other)
                    return node

                except DepthExceededException as e:
                    max_tries -= 1
                    continue
        return self

    def find_crossover_node(self, tp, is_root=False):
        nodes = []
        # first scan tree if there are prims who return the same type
        if tp == self.primitive.returns() and not is_root:
            nodes.append(self)
        for child in self.childern:
            nodes += child.find_crossover_node(tp)
        return nodes

    # TO ADD
    def crossover(self, other, prob=0.2, p=None, is_root=False):
        if random.random() < prob and not is_root:
            # do crossover here
            tp = self.primitive.returns()
            nodes = other.node.find_crossover_node(tp, is_root=True)
            if len(nodes) > 0:
                other_node = random.choice(nodes)
                other_childern = other_node.childern
                other_prim = other_node.primitive
                other_node.childern = self.childern
                other_node.primitive = self.primitive
                self.childern = other_childern
                self.primitive = other_prim
        elif len(self.childern):
            random.choice(self.childern).crossover(other, prob=prob, p=p)

    def to_program(self):
        try:
            if len(self.childern) == 0:
                return f' {self.primitive} '
            else:
                prog = f'({self.primitive} '

                for child in self.childern:
                    prog += child.to_program()

                prog += ')'
        except:
            import traceback
            traceback.print_exc()
        return prog

    def is_terminal(self):
        from dsl import Terminal
        return type(self.primitive) == Terminal

    def is_input_parameter(self):
        from dsl import InputParameter
        return type(self.primitive) == InputParameter

    def is_invented(self):
        from dsl import Invented
        return type(self.primitive) == Invented

    def evaluate(self, input_params, return_name=False):
        if self.is_terminal():
            if return_name:
                return self.primitive.name
            return self.primitive.val
        elif self.is_input_parameter():
            if return_name:
                return self.primitive.name
            return input_params[self.primitive.val]
        elif self.is_invented():
            params = [child.evaluate(input_params, return_name=True)
                      for child in self.childern]
            return self.primitive.fn(params, input_params)
        else:
            params = [child.evaluate(input_params) for child in self.childern]
            if return_name:
                try:
                    result = self.primitive.fn(*params)
                    # we need to find the correct primitive to return the name
                    return self.grammar.find_by_type_and_value(self.primitive.returns(), result).name
                except:
                    print('looked for', self.primitive.returns(), result)
                    assert False
            return self.primitive.fn(*params)

    def parse_recursive(self, ast, is_invented):
        self.childern = []
        # check that the types match
        i = 0
        for str_prim in ast:
            if isinstance(str_prim, list):
                prim = self.grammar.get_primitive(str_prim.pop(0))
            else:
                try:
                    prim = self.grammar.get_primitive(str_prim)
                    i += 1
                except:
                    # try to sample correct subtree..
                    # self.random_subtree(self.grammar, 5)
                    raise ProgramNotCorrectlyRewrittenException(
                        f'{prim} returns {prim.returns()} but expected {self.primitive.parameters()} (i {i})')

            child = Node(prim, self.grammar)
            child.childern = []
            self.childern.append(child)

            if isinstance(str_prim, list):
                child.parse_recursive(str_prim, is_invented)

    def parse_recursive_without_exception(self, ast):
        self.childern = []
        # check that the types match
        for str_prim in ast:
            try:
                if isinstance(str_prim, list):
                    prim = self.grammar.get_primitive(str_prim.pop(0))
                else:
                    prim = self.grammar.get_primitive(str_prim)
            except:
                continue

            child = Node(prim, self.grammar)
            child.childern = []
            self.childern.append(child)

            if isinstance(str_prim, list):
                child.parse_recursive_without_exception(str_prim)


class TreeRoot():
    def __init__(self, grammar):
        self.grammar = grammar
        self.node = None

    def build_random_tree(self, request, depth=6, max_tries=100, p=None):
        prim = np.random.choice(self.grammar.get_possible_childern(
            request[-1], depth, no_terminal=True), 1, p=p)[0]

        while max_tries > 0:
            try:
                node = Node(prim, self.grammar)
                node.random_subtree(depth - 1, p=p)
                break

            except DepthExceededException as e:
                max_tries -= 1
                continue
        self.node = node
        return self

    def distribution(self):
        return self.node.distribution(defaultdict(int))

    def size(self):
        return self.node.size()

    def mutate(self, p=None, prob=0.5, epsilon=0.5):
        self.node.mutate(p=p, prob=prob, is_root=True, epsilon=epsilon)

    def crossover(self, other, p=None, prob=0.2):
        self.node.crossover(other, p=p, prob=prob, is_root=True)

    def draw_tree(self, fname, footer, show=False):
        dot = [Digraph()]
        dot[0].attr(kw='graph', label=footer)
        count = [0]
        self.node.draw(dot, count)
        src = Source(dot[0], filename=fname + ".gv", format="png")
        src.render()

    def test_program(self, task):
        correct = 0
        try:
            for example in task.examples:
                output = self.node.evaluate(example[0])
                if output == example[1]:
                    correct += 1
        except Exception as e:
            # raise ProgramNotCorrectlyTypedException(self.get_program())
            print(e)
            return False

        return correct == len(task.examples)

    def get_program(self):
        prog = self.node.to_program()
        # prog = prog.replace('input-direction', '$0')
        # prog = prog.replace('input-map', '$1')
        # return f'(lam (lam {prog} ))'
        return prog

    @staticmethod
    def from_program(program, grammar, is_invented=False):
        program = str(program)
        root = TreeRoot(grammar)
        ast = parser.parse(program)
        # first of list is always name of function
        data = ast.pop(0)
        prim = grammar.get_primitive(data)
        root.node = Node(prim, grammar)
        root.node.parse_recursive(ast, is_invented)
        return root

    @staticmethod
    def calc_invented_size(program, grammar):
        program = str(program)
        root = TreeRoot(grammar)
        ast = parser.parse(program)
        # first of list is always name of function
        data = ast.pop(0)
        prim = grammar.get_primitive(data)
        root.node = Node(prim, grammar)
        root.node.parse_recursive_without_exception(ast)
        return root.size()
