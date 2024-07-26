from gym_minigrid.minigrid import MiniGridEnv, IDX_TO_OBJECT
from typing import Callable, List
from collections import defaultdict
import parser
from tree import *

idx_to_action = {int(a): a.name for a in MiniGridEnv.Actions}


class GameObject():
    def __init__(self, obj, x, y):
        self.obj = obj
        self.x = x
        self.y = y

    def __eq__(self, other) -> bool:
        return self.obj == other.obj


tdirection = "tdirection"  # the objects from the gyls gom minigrid env
tinpdirection = "tinpdirection"  # the objects from the gym minigrid env
tobj = "tobj"  # the objects from the gym minigrid env
tgameobj = "tgameobj"  # the objects from the gym minigrid env
trow = "trow"
tmap = "tmap"
taction = "taction"
t0 = "t0"
tint = "tint"
tbool = "tbool"
tendbool = "tendbool"


def _if(c, t, f): return t if c else f
def _and(x, y): return x and y
def _not(x): return not x
def _or(x, y): return x or y
def _get(m, x, y): return GameObject(m[x][y], x, y)
def _eq(x, y): return x == y
def _eq_direction(x, y): return x == y
def _eq_obj(x: GameObject, y): return x.obj == y
def _get_game_obj(obj: GameObject): return obj.obj


class Primitive():
    def __init__(self, name: str, tp: List[str], fn: Callable, val: int = None):
        self.name = name
        self.fn = fn
        self.tp = tp
        self.val = val

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name

    def returns(self):
        return self.tp[-1]

    def parameters(self):
        return self.tp[:-1]

    def is_function(self): return True

    def is_invented(self): return False


class Terminal(Primitive):
    def __init__(self, name: str, tp: str, val: int):
        super().__init__(name, [tp], None, val)

    def is_function(self): return False


class InputParameter(Terminal):
    def __init__(self, name: str, tp: str, val: int):
        super().__init__(name, tp, val)

    def is_function(self): return False


class Invented(Primitive):
    def __init__(self, name: str, body: str, arity: int, grammar):
        self.body, tp = self.infer_types(body, grammar, arity)
        fn = self.make_callable(grammar)
        self.cache = {}
        super().__init__(name, tp, fn)

    def is_invented(self): return True

    def recursive_infer(self, ast, var_name, grammar):
        # first of ast is always the function
        fn_name = ast[0]
        fn = grammar.get_primitive(fn_name)
        if var_name in ast:
            idx = ast.index(var_name) - 1
            tp = fn.parameters()[idx]
            return tp
        else:
            # search for the next functions
            for node in ast[1:]:
                if isinstance(node, list):
                    tp = self.recursive_infer(node, var_name, grammar)
                    if tp is not None:
                        return tp

    def infer_types(self, body, grammar, arity):
        tps = [None] * arity
        ast = parser.parse(body)
        for i in range(arity):
            var_name = f'#{i}'
            tp = self.recursive_infer(ast, var_name, grammar)
            tps[i] = tp

        fn = grammar.get_primitive(ast[0])
        level_arity = len(ast[1:])

        # check if we need to add parameters
        while level_arity < len(fn.parameters()):
            body = f"{body[:-1]} #{arity})"
            tps.append(fn.parameters()[level_arity])
            level_arity += 1
            arity += 1

        # this only holds if arity is the same for first function and invented
        fn = grammar.get_primitive(ast[0])
        tps.append(fn.returns())
        return body, tps

    def make_callable(self, grammar):
        def execute(params, input_params):
            body = self.body

            for i, param in enumerate(params):
                body = body.replace(f'#{i}', str(param))
            if body in self.cache:
                return self.cache[body]
            else:
                result = TreeRoot.from_program(
                    body, grammar, is_invented=True).node.evaluate(input_params)
                self.cache[body] = result
                return result
        return execute


class Grammar():
    def __init__(self, inputs: List[InputParameter], terminals: List[Terminal], primitives: List[Primitive], max_invented_length=6):
        self.primitives = primitives
        self.inputs = inputs
        self.terminals = terminals
        self.mapping = self.map_primitives_to_types(
            primitives + inputs + terminals)
        self.invented = []
        self.dict = {p.name: p for p in self.productions}
        self.max_invented_length = max_invented_length

    @property
    def productions(self):
        return self.inputs + self.primitives + self.terminals + self.invented

    def map_primitives_to_types(self, primitives):
        mapping = defaultdict(lambda: [])
        for p in primitives:
            mapping[p.returns()].append(p)
        return mapping

    def get_possible_childern(self, tp: str, depth: int, no_terminal=False):
        if depth <= 0:
            # enforce terminal if possible
            only_terminals = [prim for prim in self.mapping[tp]
                              if len(prim.parameters()) == 0]
            if len(only_terminals) > 0:
                return only_terminals
        elif no_terminal:
            only_functions = [prim for prim in self.mapping[tp]
                              if len(prim.parameters()) > 0]
            if len(only_functions) > 0:
                return only_functions
        return self.mapping[tp]

    def add_invented(self, name: str, body: str, arity: int):
        assert name not in self.dict.keys()

        size = TreeRoot.calc_invented_size(body, self)
        if size > self.max_invented_length:
            print(
                f'Invented {name} was not added because {size} > {self.max_invented_length}')
            print(body)
            return False

        new_invented = Invented(name, body, arity, self)
        self.dict[name] = new_invented
        self.mapping[str(new_invented.returns())].append(new_invented)
        self.invented.append(new_invented)
        return True

    def get_primitive(self, name):
        return self.dict[str(name)]

    def find_by_type_and_value(self, tp, val):
        if tp == tgameobj:
            tp = tobj
        elif tp == tinpdirection:
            tp = tdirection
        elif tp == tbool:
            tp = tendbool
        # print('look for', str(tp), val)
        for t in self.terminals:
            # print('check', t, tp, val)
            if str(t.returns()) == str(tp) and val == t.val:
                return t


def base_primitives():
    return [
        InputParameter('$1', tmap, 0),
        InputParameter('$0', tinpdirection, 1)
    ], [
        Terminal(str(j), tint, j) for j in range(5)
    ] + [
        Terminal(f'{action}-action', taction, idx) for idx, action in idx_to_action.items() if action in ['left', 'right', 'forward']
    ] + [
        Terminal(f'direction-{idx}', tdirection, idx) for idx in range(4)
    ] + [
        Terminal(f'{obj}-obj', tobj, idx) for idx, obj in IDX_TO_OBJECT.items() if obj in ['empty', 'wall', 'goal']
    ] + [
        Terminal('True', tendbool, True),
        Terminal('False', tendbool, False)
    ], [
        Primitive("if_action", (tbool, taction, taction, taction), _if),
        Primitive("if_direction", (tbool, tdirection,
                  tdirection, tdirection), _if),
        Primitive("if_object", (tbool, tobj, tobj, tobj), _if),
        # Primitive("if_bool", (tbool, tbool, tbool, tbool), _if),
        Primitive("if_int", (tbool, tint, tint, tint), _if),
        Primitive("eq-direction?", (tinpdirection,
                  tdirection, tbool), _eq_direction),
        Primitive("eq-obj?", (tgameobj, tobj, tbool), _eq_obj),
        Primitive("get", (tmap, tint, tint, tgameobj), _get),
        Primitive("not", (tbool, tbool), _not),
        Primitive("and", (tbool, tbool, tbool), _and),
        Primitive("or", (tbool, tbool, tbool), _or),
        Primitive("get-game-obj", (tgameobj, tobj), _get_game_obj),
    ]


g = Grammar(*base_primitives())
