# Similitud semántica FINTERM: Qwen3-Embedding-4B y MAX

## Modelo y distinción metodológica

Se adopta `Qwen/Qwen3-Embedding-4B`, revisión fija **`5cf2132abc99cad020ac570b19d031efec650f2b`**. La [ficha oficial fijada](https://huggingface.co/Qwen/Qwen3-Embedding-4B/blob/5cf2132abc99cad020ac570b19d031efec650f2b/README.md) publica **88,72 en STS de MTEB English v2**. Es la justificación de la selección inicial, no una medida obtenida sobre nuestro enlazador.

**Se sustituye el antiguo componente por un encoder semántico moderno; matemáticamente se sigue usando coseno**, como recomienda el propio modelo. En concreto se llama a `sentence_transformers.util.cos_sim`, la misma función oficial usada por `SentenceTransformer.similarity` con `similarity_fn_name="cosine"`. No se presenta el producto de embeddings normalizados como una eliminación del coseno.

La configuración ST oficial es Transformer → **last-token Pooling** → Normalize, con dimensión completa **2560**. Se carga esa cadena nativa; no se implementa otro pooling. Se utiliza bfloat16, padding izquierdo y SDPA de PyTorch, sin instalar FlashAttention ni cuantizar este nuevo modelo a NF4. La cuantización de Qwen3-14B permanece intacta.

Perfil inicial **`SYMMETRIC_STS_EMPTY_PROMPT_V1`**: ambos lados son textos de comparación STS, sin prompt de búsqueda. El repositorio oficial tiene `default_prompt_name=null` y `similarity_fn_name="cosine"`. Su prompt opcional `query` sirve para retrieval/web-QA; aquí se pasa explícitamente `prompt=""` en ambos lados. Esta elección se registra para poder confirmarla con el equipo investigador; no se introduce silenciosamente una instrucción de retrieval. Los embeddings se normalizan mediante Sentence Transformers y se conservan normalizados en float32 en caché.

## Archivos y contrato

- `finterm_similarity.py`: inputs, integridad, preparación, caché, scores individuales, MAX y diagnóstico de referencias.
- `finterm_similarity_qwen.py`: preflight, recursos fijados, encoder ST oficial y coseno nativo.
- `tests/test_similarity.py`, `tests/run_similarity_offline.py`: aceptación sin modelos/red.
- `requirements.txt`: añade únicamente `sentence-transformers==5.2.0` al conjunto ya fijado; conserva los pins CUDA/Qwen.

La entrada principal es el directorio de salida de DefTerm, con `05_defterm_manifest.json` (`finterm-defterm-v1`) y sus archivos declarados. Se verifican todos los hashes y se consumen:

| Contrato | Uso |
|---|---|
| `05_defterm_occurrences.jsonl` | Contexto exacto y UID de cada aparición, heredados de Sentido 1. |
| `05_defterm_associations.jsonl` | Asociaciones document_id/term_id/occurrence_uid/candidate_id/QID de fase 02. |
| `05_defterm_candidates.jsonl` | Candidatos originales y sus `all_representation_ids`, incluyendo definiciones. |
| `05_defterm_representations.jsonl` | Descripciones inglesas, traducciones y definiciones disponibles, con sus estados. |
| `05_defterm_terms.jsonl` | Todos los estados originales, incluso términos no encontrados. |

Los outputs de DefTerm son autocontenidos respecto de contextos/asociaciones. No se vuelve a leer el texto por una ruta externa ni se recalculan contextos/candidatos/representaciones. Si están disponibles, `stage01_dir=` y `stage02_dir=` permiten contrastarlos adicionalmente con los contratos originales usando los lectores existentes. La procedencia distingue explícitamente copia verificada en DefTerm de comprobación adicional contra 01/02.

Se admiten runs parciales/preparados/interrumpidos; una traducción pendiente no desaparece ni se confunde con un score bajo. La preselección nominal de DefTerm **no filtra la similitud**: se comparan todos los candidatos asociados que tengan representaciones inglesas disponibles, también secundarios con texto.

## Comparación y ranking

Para cada asociación de una aparición con un candidato:

1. Se codifica exclusivamente `context`, sin marcadores, concatenación de otras apariciones, label adicional ni QID.
2. Se codifica cada `text_en` cuyo `available_for_later_evaluation=true`. Se conservan textos de una palabra, variantes inglesas, traducciones y DefTerm. No se inventa una representación a partir del label si falta texto.
3. Se calcula y guarda el coseno de **cada par aparición–candidato–representación**.
4. Se agrega **MAX exclusivamente dentro de esa aparición/candidato**. No se promedian ni se agrupan contextos de apariciones diferentes.
5. Se ordena de mayor a menor score. Se guardan todos los IDs de representación que empatan en el máximo y un ID representativo para mostrar, elegido de forma determinista. Los empates de candidatos se ordenan por QID numérico/candidate_id solo para presentación y conservan el mismo `score_rank`.

No hay umbral semántico, cross-encoder, segundo reranker ni elección de enlace. Un ranking puede tener solo scores negativos: se conserva igualmente. El coseno no es una probabilidad ni una prueba de identidad conceptual. MAX puede favorecer a candidatos con más representaciones; se registran cantidades disponibles/puntuadas/pendientes para observarlo, sin introducir otra agregación o corrección oculta.

Los candidatos sin texto tienen `score=null`, `rank_position=null` y un estado explícito. Si tienen algún texto utilizable y otras traducciones pendientes, se puntúa lo disponible y se conserva la cobertura incompleta. Los términos fuera del piloto quedan `NOT_IN_PILOT_SCOPE`; sus candidatos y datos originales siguen en los outputs.

## Caché, presupuesto e integridad

`run_similarity(..., max_new_embeddings=0)` prepara por defecto sin cargar pesos. El presupuesto cuenta textos nuevos distintos; las coincidencias de caché no lo consumen. Una cadena exactamente igual puede reutilizar su embedding entre varios QIDs/idiomas o apariciones, pero cada score/ranking sigue vinculado a sus IDs propios: **reutilizar un vector no combina contextos**.

La caché SQLite propia almacena vectores float32 little-endian y SHA-256, request de texto/modelo y metadatos reales. No usa pickle. Valida dimensión, finitud y normalización; la tolerancia numérica de norma no es un umbral de aceptación de enlaces. El request incluye revisión/configuración/perfil; una caché de otro modelo o una producción sintética no se reutiliza como Qwen real.

Se confirma cada vector terminado sin reescribir todo el historial. Tras una interrupción se mantienen los vectores confirmados; una ejecución posterior con presupuesto positivo puede recalcular solo los que no llegaron a confirmarse. No se reintentan automáticamente fallos dentro de la misma ejecución. Un input que supera **32768 tokens** queda `INPUT_TOO_LONG`, sin truncarlo silenciosamente, y se conservan sus candidatos/representaciones. Si otros textos de ese batch son válidos, se procesan separadamente.

La caché y outputs deben estar fuera de inputs y en directorios explícitos nuevos. Las comprobaciones de hash detectan incoherencia/corrupción; no son firmas de autenticidad frente a alguien que reescriba todos los archivos y hashes.

## Recursos y coexistencia GPU

Los pesos son aproximadamente **8,04 GB**, en dos shards, más tokenizer/configuración. Se preparan una sola vez en almacenamiento persistente **fuera del código**. El índice upstream del commit fijado está en el adaptador: tamaños, SHA-256 para archivos LFS y digest de blob para archivos pequeños. El manifiesto local añade SHA-256 de todos los archivos. La revisión y la cadena Pooling/Normalize se verifican antes de utilizar los recursos.

`provision_resources(...)` es la única operación que puede descargar pesos. Reutiliza un snapshot completo de la caché hub si se indica `hub_cache_dir`; si ya existe el manifiesto válido, no hace llamadas de descarga. `run_similarity` y el encoder no invocan esa operación y siempre cargan con `local_files_only=True`, `trust_remote_code=False`.

Antes de construir SentenceTransformer se comprueban CUDA/H100/bfloat16, memoria libre global y memoria asignada/reservada por el proceso. El presupuesto de memoria es tamaño de pesos más una reserva configurable (8 GiB por defecto para activaciones, batch inicial 1); no garantiza que cualquier longitud/batch quepa. Por defecto se bloquea la carga si el proceso ya tiene memoria GPU asignada/reservada: usar un kernel limpio para evitar coexistir accidentalmente con Qwen3-14B. `allow_shared_gpu=True` es explícito y mantiene la comprobación de memoria libre; no se descargan ni destruyen otros modelos automáticamente.

`backend.close()` libera solo el encoder de esta fase después de la prueba. El cálculo de coseno sobre vectores cacheados puede hacerse en CPU sin volver a cargar los pesos.

## 1. Preflight antes de instalar o descargar

En H100, con el directorio de código actualizado, ejecutar primero:

```python
from pathlib import Path
import sys, json
CODE = Path("/RUTA_REAL_DEL_CODIGO/terminlink_en")
sys.path.insert(0, str(CODE))
from finterm_similarity_qwen import check_environment
print(json.dumps(check_environment(), indent=2))
```

Se requiere Python 3.11 y el conjunto validado: torch `2.10.0+cu126`, Transformers `4.57.6`, Accelerate `1.15.0`, bitsandbytes `0.50.2`, y Sentence Transformers **5.2.0**. La consulta solo lee metadata; no importa Torch/ST ni inicia CUDA. No se afirma haber inspeccionado a distancia el entorno H100 desde el host Windows.

Sentence Transformers 5.2.0 declara Python >=3.10, Transformers >=4.41,<6 y torch >=1.11: incluye el entorno validado y el mínimo Transformers >=4.51 del modelo. ST 6.0.0 exige Transformers >=5 y no es compatible con mantener 4.57.6. Si falta ST o es incompatible, revisar primero la resolución desde el entorno del kernel, en el directorio de código:

```text
python -m pip install --dry-run -r requirements.txt
python -m pip install -r requirements.txt
python -m pip check
```

Los pins CUDA/Qwen permanecen en `requirements.txt`; no se recomienda `--upgrade`, `--force-reinstall` ni instalar una versión de Torch genérica. Reiniciar el kernel tras cambiar paquetes y repetir el preflight. El runtime no ejecuta pip.

## 2. Aprovisionamiento explícito una sola vez

En un kernel de preparación con acceso permitido a Hugging Face (sin modelos GPU), tras preflight compatible:

```python
import os
os.environ["HF_HUB_OFFLINE"] = "0"  # Antes de importar huggingface_hub en este kernel de setup.
os.environ["TRANSFORMERS_OFFLINE"] = "0"
from finterm_similarity_qwen import provision_resources
RESOURCES = Path("/ALMACENAMIENTO_PERSISTENTE_EXISTENTE/qwen3_embedding4b")
manifest = provision_resources(RESOURCES)
# Si el modelo YA existe en otra caché hub persistente:
# manifest = provision_resources(RESOURCES, hub_cache_dir=Path("/CACHE_HUB_EXISTENTE"))
print(manifest["model_id"], manifest["model_revision"], manifest["content_sha256"])
```

No ejecutar esta celda por cada ranking. Reservar espacio persistente suficiente para los shards y temporales de descarga; no incluirlos en la entrega de código. En esta implementación local no se han descargado los pesos.

## 3. Celda exacta del primer piloto H100

Usar un **kernel limpio** que no tenga Qwen3-14B residente. Sustituir las cuatro rutas, mantener todos los candidatos recuperados y ejecutar solo cuando el setup anterior esté terminado:

```python
# PRIMER_PILOTO_REAL_H100
import os, sys, json
from pathlib import Path
from datetime import datetime, timezone
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

CODE = Path("/RUTA_REAL_DEL_CODIGO/terminlink_en")
DEFTERM = Path("/RUTA_REAL_OUTPUT_DEFTERM")
RESOURCES = Path("/ALMACENAMIENTO_PERSISTENTE_EXISTENTE/qwen3_embedding4b")
WORK = Path("/RUTA_PERSISTENTE_EXISTENTE/piloto_similitud")
assert CODE.is_dir() and DEFTERM.is_dir() and WORK.parent.is_dir()
assert (RESOURCES / "embedding_resources.json").is_file(), "Preparar una sola vez los recursos fijados"
assert WORK.resolve() != CODE.resolve() and WORK.resolve() != DEFTERM.resolve()
assert DEFTERM.resolve() not in WORK.resolve().parents and WORK.resolve() not in DEFTERM.resolve().parents
WORK.mkdir(exist_ok=True)
sys.path.insert(0, str(CODE))
from finterm_similarity_qwen import check_environment, QwenEmbedding4B
from finterm_similarity import load_similarity_input, prepare_similarity, run_similarity, reference_positions

environment = check_environment()
print(json.dumps(environment, indent=2))
assert environment["compatible"], "Resolver dependencias antes de cargar pesos; no actualizar Torch/CUDA/Qwen"
terms = ["interest rates", "Commercial banks", "basis points", "European Central Bank"]
source = load_similarity_input(DEFTERM)
# Si se conservan sus rutas, se pueden añadir stage01_dir=... y stage02_dir=...
preview = prepare_similarity(source, terms=terms)
print("Plan:", preview["counts"])
assert preview["counts"]["unique_embedding_requests"] <= 4096, "Revisar el alcance del piloto antes de inferir"
session = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
output = WORK / session
backend = QwenEmbedding4B(RESOURCES, batch_size=1)
try:
    result = run_similarity(source, output, WORK / "qwen3_embedding4b_vectors.sqlite",
        backend=backend, terms=terms,
        max_new_embeddings=preview["counts"]["unique_embedding_requests"])
finally:
    backend.close()
print(result["status"], result["counts"], result["execution"])
if result["execution"]["backend_error"]:
    raise RuntimeError(result["execution"]["backend_error"])
references = {"interest rates": "Q179179", "Commercial banks": "Q848507",
              "basis points": "Q750178", "European Central Bank": "Q8901"}
positions = reference_positions(output, references)
print(json.dumps(positions, ensure_ascii=False, indent=2))
print("Artefactos:", output)
source.verify()
```

Los cuatro QIDs se utilizan **después** de puntuar para observar su posición por aparición. No entran al encoder, no alteran el orden ni se consideran un gold completo. No se calculan métricas globales de exactitud y no se fuerza que ocupen el primer puesto. Si falta una referencia en el inventario recuperado se informa, sin consultar Wikidata ni insertarla artificialmente.

El límite 4096 de la celda es un presupuesto operativo del piloto, no un umbral de similitud. Los candidatos de otros términos siguen conservados y fuera del alcance de puntuación. Para repetir se usa otro output y la misma caché; con `max_new_embeddings=0` no se generan nuevos vectores.

## Outputs

- `06_similarity_scores.jsonl`: score coseno individual, IDs de aparición/candidato/QID/representación, origen/idioma, hashes y estado.
- `06_candidate_rankings.jsonl`: MAX, posición, grupo de empate, todos los IDs que producen el máximo, cantidades y estados originales.
- `06_occurrence_status.jsonl`: contextos originales y cobertura de puntuación por aparición, incluso sin candidatos.
- `06_embedding_status.jsonl`: vectores disponibles, pendientes, largos o fallidos, con hashes de payload.
- `06_input_candidates.jsonl`, `06_input_representations.jsonl`, `06_input_terms.jsonl`, `06_input_associations.jsonl`: datos originales conservados.
- `06_similarity_source.json`, `06_similarity_manifest.json`: autoridad de inputs, hashes, modelo/revisión, perfil, ejecución y alcance.

`COMPLETED` significa que terminó el cálculo solicitado, no que la recuperación/las traducciones upstream sean completas ni que exista un enlace correcto. Se conserva `source_status`. `PREPARED`, `PARTIAL` e `INTERRUPTED` distinguen pendientes de embeddings, errores y cancelaciones. Las salidas de error no contienen mensajes arbitrarios de excepciones que puedan revelar configuración externa.

## Pruebas offline

```text
python -B tests/run_similarity_offline.py
```

El runner bloquea red e imports ML reales. Comprueba álgebra coseno controlada, MAX por aparición, empates, scores negativos, monopalabras, definiciones generadas, reutilización, corrupción, límites de tokens, interrupciones, referencias y contratos originales 01/02. También prepara el ZIP real de 280 candidatos sin generar embeddings reales ni presentar rankings sintéticos como resultados de Qwen.

### Comprobaciones realizadas en esta entrega

- **32 pruebas de similitud offline**, sin fallos/omisiones, incluida ejecución de la celda exacta H100 con encoder controlado y el ZIP real como preparación sin inferencia.
- **64 pruebas DefTerm y 47 de Sentido 2** superadas como regresión; esos componentes no se modifican.
- Instalación de ST **5.2.0** en un entorno **aislado local Windows/Python 3.12.2**, conservando torch **2.10.0+cu126**, CUDA compilada **12.6**, Transformers **4.57.6**, Accelerate **1.15.0** y bitsandbytes **0.50.2**. `pip check` sin conflictos. No es una instalación o inspección remota de H100.
- Verificación adicional de `sentence_transformers.util.cos_sim` real sobre tensores CPU de prueba: resultados 1, 0 y -1 correctos, sin instanciar modelos ni descargar pesos.
- Preparación del piloto real de cuatro términos: **280 candidatos**, **5 apariciones de origen** (la de `mortgage rates` queda fuera de alcance), **3.508 representaciones originales**, **290 pares individuales** y **189 textos únicos a codificar**. Se conservan todos los datos; las cuatro referencias aparecen con rango/score pendientes, no como resultados reales.
- Artefactos de esa preparación, con cero inferencias y sin imports ML, en `<LOCAL_WORKSPACE>`.

La comprobación real de pesos, memoria, embeddings y rankings H100 queda pendiente de ejecutar las celdas de setup/piloto anteriores. El valor MTEB y los controles matemáticos no se presentan como resultados de enlazado.
