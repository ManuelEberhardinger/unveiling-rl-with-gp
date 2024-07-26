import re


def tokenize(txt):
    tokens = re.split('(\s+|\(|\))', txt)
    return [t for t in tokens if len(t) and not t.isspace()]


def parse(txt):
    tokens = tokenize(txt)
    ast, tokens = parse_list(tokens)
    if len(tokens) > 0:
        raise SyntaxError(
            "(parse) Error: not all tokens consumed <%s>" % str(tokens))

    return ast


def parse_list(tokens):
    if len(tokens) == 0 or tokens[0] != '(':
        raise SyntaxError(
            "Error: expected '(' token, found <%s>" % str(tokens))
    first = tokens.pop(0)

    operator = tokens.pop(0)
    operands, tokens = parse_operands(tokens)
    ast = [operator]
    ast.extend(operands)

    if len(tokens) == 0 or tokens[0] != ')':
        raise SyntaxError(
            "Error: expected ')' token, found <%s>: " % str(tokens))
    first = tokens.pop(0)

    return ast, tokens


def parse_operands(tokens):
    operands = []
    while len(tokens) > 0:
        if tokens[0] == ')':
            break

        if tokens[0] == '(':
            subast, tokens = parse_list(tokens)
            operands.append(subast)
            continue

        operand = tokens.pop(0)
        operands.append(decode_operand(operand))

    return operands, tokens


def decode_operand(token):
    if is_int(token):
        return int(token)
    elif is_float(token):
        return float(token)
    else:  # default to a string
        return str(token)


def is_float(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def is_int(s):
    try:
        int(s)
        return True
    except ValueError:
        return False


if __name__ == "__main__":
    s = '(lambda (lambda (if (and (eq-direction? $0 direction-2) (eq-obj? (get $1 0 0) m-empty-obj)) m-left-action m-forward-action)))'
    parse_list(tokenize(s))[0]
