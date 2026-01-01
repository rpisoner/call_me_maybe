from llm_sdk import Small_LLM_Model
import json
import argparse


def build_response(model: Small_LLM_Model, prompt: str) -> str:
    """Genera una respuesta del modelo dado un prompt."""
    tokens = model.encode(prompt)
    inicio_respuesta = len(tokens)
    FINAL_TOKEN = model.get_eos_token_id()

    for i in range(50):
        logits = model.get_logits_from_input_ids(tokens)
        nuevo_token = logits.index(max(logits))
        tokens.append(nuevo_token)

        if nuevo_token == FINAL_TOKEN:
            break

    respuesta = model.decode(tokens[inicio_respuesta:])
    return respuesta


def load_json_file(filepath: str) -> list | None:
    """Carga un archivo JSON y devuelve su contenido."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo {filepath}")
        return None
    except json.JSONDecodeError:
        print(f"Error: El archivo {filepath} no es un JSON válido")
        return None


def parse_arguments() -> argparse.Namespace:
    """Procesa los argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(description="JSON AI")
    parser.add_argument("--input",
                        default="data/input/",
                        help="Ruta de entrada")
    parser.add_argument("--output",
                        default="data/output/function_calling_results.json",
                        help="Ruta de salida")
    return parser.parse_args()


def main() -> None:
    print("Cargando modelo...")
    model = Small_LLM_Model()
    print("¡Modelo cargado!")

    args = parse_arguments()

    input_dir = args.input
    tests_file = input_dir + "function_calling_tests.json"
    functions_file = input_dir + "functions_definition.json"
    output_file = args.output

    tests = load_json_file(tests_file)
    if tests is None:
        return

    functions = load_json_file(functions_file)
    if functions is None:
        return

    resultados = []
    for test in tests:
        prompt = test["prompt"]
        respuesta = build_response(model, prompt)

        # RESULTADO PARA ORGANIZAR RESPUESTA, SERA UTIL MAS ADELANTE
#        resultado = {
#            "prompt": prompt,
#            "fn_name": "???",
#            "args": {}
#        }

        resultados.append(respuesta)
        print("--PROMPT--")
        print(prompt)
        print("--RESPUESTA--")
        print(respuesta, "\n")

    # Al final, guardar todo
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(resultados, f, indent=2)
