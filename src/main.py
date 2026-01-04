from llm_sdk import Small_LLM_Model
import json
import argparse

def filtrar_logits(logits: list[float], tokens_permitidos: list[int]) -> list[float]:
    """Pone -inf en todos los tokens excepto los permitidos."""
    logits_filtrados = [float('-inf')] * len(logits)

    for token_id in tokens_permitidos:
        logits_filtrados[token_id] = logits[token_id]

    return logits_filtrados


def generar_nombre_funcion(
    model: Small_LLM_Model,
    tokens: list[int],
    functions: list
) -> tuple[list[int], str]:
    """
    Genera el nombre de la función con decodificación restringida.
    Devuelve: (tokens actualizados, nombre_elegido)
    """
    # Obtener nombres de funciones válidas
    nombres_validos = []
    for f in functions:
        nombres_validos.append(f["fn_name"])
    
    # Tokenizar cada nombre
    tokens_por_nombre = {}
    for nombre in nombres_validos:
        tokens_por_nombre[nombre] = model.encode(nombre)

    # Variables para rastrear el estado
    nombres_posibles = nombres_validos.copy()
    posicion = 0
    nombre_elegido = ""

    # Bucle de generación del nombre
    for i in range(50):
        logits = model.get_logits_from_input_ids(tokens)

        # Obtener tokens permitidos en la posición actual
        tokens_permitidos = []
        for nombre in nombres_posibles:
            lista_tokens = tokens_por_nombre[nombre]
            if posicion < len(lista_tokens):
                tokens_permitidos.append(lista_tokens[posicion])

        # Filtrar y elegir
        logits_filtrados = filtrar_logits(logits, tokens_permitidos)
        nuevo_token = logits_filtrados.index(max(logits_filtrados))
        tokens.append(nuevo_token)

        # Filtrar nombres que ya no son posibles
        nombres_siguientes = []
        for nombre in nombres_posibles:
            lista_tokens = tokens_por_nombre[nombre]
            if posicion < len(lista_tokens) and lista_tokens[posicion] == nuevo_token:
                nombres_siguientes.append(nombre)
        nombres_posibles = nombres_siguientes
        posicion += 1

        # Si solo queda un nombre y completamos sus tokens, parar
        if len(nombres_posibles) == 1:
            nombre_elegido = nombres_posibles[0]
            if posicion >= len(tokens_por_nombre[nombre_elegido]):
                break

        # Parada de emergencia
        if len(nombres_posibles) == 0:
            break

    return tokens, nombre_elegido


def build_response(
    model: Small_LLM_Model,
    prompt: str,
    functions: list
) -> dict:
    """
    Función principal que orquesta la generación de la respuesta JSON.
    Devuelve: diccionario con fn_name y args
    """
    # Paso 1: Construir contexto con funciones disponibles
    contexto = "Available functions:\n"
    for f in functions:
        nombre = f["fn_name"]
        args = ", ".join(f["args_names"])
        contexto += f"- {nombre}({args})\n"
    contexto += "\nCall the appropriate function for the user request.\n\n"
    
    # Paso 2: Preparar prompt completo
    prefijo = '{"fn_name": "'
    prompt_completo = contexto + "User: " + prompt + '\nFunction call: ' + prefijo
    tokens = model.encode(prompt_completo)

    # Paso 3: Generar nombre de función
    tokens, nombre_elegido = generar_nombre_funcion(model, tokens, functions)

    # Paso 4: TODO - Generar argumentos (por implementar)
    args = {}

    # Devolver resultado estructurado
    return {"fn_name": nombre_elegido, "args": args}


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
    vocab_path = model.get_path_to_vocabulary_json()
    vocab = load_json_file(vocab_path)
    if vocab is None:
        return
    vocab_inv = {v: k for k, v in vocab.items()}

    tests = load_json_file(tests_file)
    if tests is None:
        return

    functions = load_json_file(functions_file)
    if functions is None:
        return

    resultados = []
    for test in tests:
        prompt = test["prompt"]
        respuesta = build_response(model, prompt, functions)

        resultado = {
            "prompt": prompt,
            "fn_name": respuesta["fn_name"],
            "args": respuesta["args"]
        }

        print("\n--PROMPT--\n", prompt, "\n")
        print("--RESPONSE--\n", resultado)
        resultados.append(resultado)

    # Al final, guardar todo
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(resultados, f, indent=2)
