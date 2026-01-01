*Este proyecto ha sido creado como parte del currículo de 42 por rpcla.*

# Call Me Maybe - Function Calling in LLMs

## Descripción
Este proyecto implementa una herramienta de llamada a funciones para LLMs utilizando **decodificación restringida**. El objetivo es traducir peticiones en lenguaje natural a llamadas de función estructuradas (JSON) que sean 100% válidas y cumplan con un esquema predefinido, incluso utilizando un modelo pequeño (Qwen 0.5B).

## Instrucciones

### Instalación
```bash
make install
```

### Ejecución
```bash
make run
```
Para usar archivos de entrada/salida personalizados:
```bash
uv run python -m src --input mi_input.json --output mi_output.json
```

### Linting
```bash
make lint
```

## Recursos
- **Constraint Decoding**: Técnica para limitar los tokens que el modelo puede generar basándose en una gramática o esquema.
- **LLM SDK**: Wrapper proporcionado para interactuar con el modelo.

## Explicación del Algoritmo
(A completar)

## Decisiones de Diseño
(A completar)

## Análisis de Rendimiento
(A completar)

## Retos Encontrados
(A completar)

## Estrategia de Pruebas
(A completar)
