#!/usr/bin/env python3
import sys

def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: inspect_onnx_io.py /path/to/model.onnx")
        return 2

    path = sys.argv[1]

    try:
        import onnx
    except Exception as e:
        print("ERROR: onnx python package not installed:", e)
        return 3

    m = onnx.load(path)

    inp_names = {i.name for i in m.graph.input}
    init_names = {t.name for t in m.graph.initializer}
    required = sorted(inp_names - init_names)

    print("model:", path)
    print("ir_version:", m.ir_version)
    print("opset_import:", [(op.domain or "ai.onnx", op.version) for op in m.opset_import])

    ops = sorted({n.op_type for n in m.graph.node})
    print("num_nodes:", len(m.graph.node))
    print("op_types:", ops)

    print("graph_inputs:", len(m.graph.input), "(required_feeds:", len(required), ")")
    if required:
        print("required_feed_inputs:")
        for name in required:
            print(" -", name)

    print("inputs(detail):")
    for i in m.graph.input:
        tt = i.type.tensor_type
        shape = [(d.dim_param or d.dim_value) for d in tt.shape.dim]
        print(" -", i.name, onnx.TensorProto.DataType.Name(tt.elem_type), shape,
              "(initializer)" if i.name in init_names else "")

    print("outputs:")
    for o in m.graph.output:
        tt = o.type.tensor_type
        shape = [(d.dim_param or d.dim_value) for d in tt.shape.dim]
        print(" -", o.name, onnx.TensorProto.DataType.Name(tt.elem_type), shape)

    # quick flags for ORT incompatibility
    flagged = [n for n in m.graph.node if n.op_type.lower() in ("correlation", "gridsampler", "grid_sampler", "grid_sampler_2d")]
    if flagged:
        print("\nWARNING: model contains ops often unsupported by ORT:")
        for n in flagged[:20]:
            print(" -", n.op_type, "name=", (n.name or "<no_name>"), "domain=", (n.domain or "ai.onnx"))
        if len(flagged) > 20:
            print(" - ... (", len(flagged), "total )")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
