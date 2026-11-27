# Instalación reproducible y recursos

## 1. Entorno sin GPU: contratos e instalación

Python 3.11 o 3.12; se probó localmente con Windows/Python 3.12.2.
Utilizar un entorno propio, sin modificar paquetes globales:

```bash
python -m venv .venv
# Activar .venv según el sistema operativo.
python -m pip install .
finterm-link doctor
python -B -m unittest discover -s tests -v
```

La biblioteca base no depende de Torch. Las pruebas usan solo datos sintéticos,
backends controlados identificados y, opcionalmente, un archivo real externo.
Ninguna prueba ejecuta modelos. La falta de H100 aparece en el informe doctor,
no se oculta mediante un backend falso.

## 2. Recuperación online

```bash
python -m pip install ".[wikidata]"
python -m pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl
```

El recurso de lematización original es `en_core_web_sm==3.8.0`, con spaCy 3.8.18.
La instalación del modelo lingüístico es explícita, no ocurre en `link`.
Se requiere contacto Wikimedia real mediante `--contact`, `http_contact` o
`FINTERM_WIKIMEDIA_CONTACT`. No introducir tokens o credenciales en JSON públicos.

En offline se necesitan las respuestas de caché y los lemas registrados para
cada término activo. Un cache miss es error de recuperación, no ausencia de QID.

## 3. Entorno H100

Usar Python **3.11**, PyTorch CUDA 12.6 y H100 con BF16. El perfil original se
ejecutó con Python 3.11.4. Instalar en un entorno separado:

```bash
python -m pip install -r requirements/h100-cu126.txt
python -m pip install .
```

El archivo fija las versiones directas originales de Torch, Transformers,
Sentence Transformers, Accelerate, bitsandbytes, hub, safetensors y spaCy. Se fija
tokenizers 0.22.2, compatible y observado localmente; la combinación completa
distribuida todavía necesita la validación H100. No se afirma un lock transitivo
Linux probado: el notebook conserva `pip freeze` del entorno realmente utilizado.
ST 6.x y Transformers 5.x no pertenecen a este perfil.

Seleccionar en Jupyter el kernel correspondiente al entorno preparado y comprobar `sys.executable`.

## 4. Recursos originales

| Uso | Modelo | Revisión |
|---|---|---|
| Traducción y DefTerm NF4 | `Qwen/Qwen3-14B` | `40c069824f4251a91eefaf281ebe4c544efd3e18` |
| Embeddings ST oficial | `Qwen/Qwen3-Embedding-4B` | `5cf2132abc99cad020ac570b19d031efec650f2b` |

`qwen14_hub_cache` debe ser el directorio Hugging Face que contiene
`models--Qwen--Qwen3-14B/snapshots/<revisión>`, con tokenizer, configuraciones,
índice y shards completos. `embedding_resource_dir` debe contener el manifiesto
`embedding_resources.json` y resolver el snapshot original completo. Si su
manifiesto utiliza un snapshot externo, esa ubicación debe existir en el nuevo
servidor; no se interpreta una ruta histórica como una ruta actual.

Si los recursos ya existen, verificar sin cargar modelos:

```bash
finterm-link doctor --require-h100 --hub-cache /CACHE_HF_EXISTENTE --embedding-resources /RECURSO_EMBEDDING_EXISTENTE
```

Para aprovisionar recursos por primera vez, usar explícitamente en el servidor:

```bash
python -B scripts/provision_resources.py --qwen14-hub-cache /ALMACENAMIENTO_EXTERNO/qwen14_hub --embedding-resources /ALMACENAMIENTO_EXTERNO/qwen3_embedding4b
```

Este paso separado puede descargar las revisiones fijadas y requiere espacio
persistente. Se prohíbe ejecutarlo en Windows y guardar los pesos dentro del
repositorio. La inferencia posterior siempre es `local_files_only=True` y no
llama al aprovisionamiento.

## 5. Memoria y persistencia

- Un proceso 14B para 03, otro para gate/definición 05 y otro para 06, secuenciales.
- El 14B usa NF4/doble cuantización/BF16 según su receta, con las semillas
  originales distintas de traducción y DefTerm.
- Embedding-4B conserva last-token Pooling → Normalize, BF16, SDPA, prompt vacío
  simétrico, batch 1, reserva de 8 GiB y rechazo de GPU ocupada en ese proceso.
- `gpu_lock` puede señalar una ubicación común a otros ejecutores cooperantes.
  La terminación del worker no libera modelos retenidos por otro proceso.
- `full` conserva cachés exactas de traducción, gate/definición y embeddings;
  `quick` aísla las de enriquecimiento. No hay reintento automático de errores
  ya comprometidos o claims ambiguos, como en los módulos originales.

Configurar recursos, cachés y outputs fuera del checkout, en almacenamiento
persistente. Repetir una llamada crea un run nuevo y reutiliza las cachés
compatibles; nunca sobrescribe un resultado anterior.
