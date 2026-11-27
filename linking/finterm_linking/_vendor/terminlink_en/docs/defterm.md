# DefTerm dinámico: gate de suficiencia y definición few-shot

## Integración y archivos necesarios

La entrada es un directorio comprometido por `03_manifest.json`, producido por el nuevo Sentido 2 (`finterm-sense2-v1`). Se admiten estados `COMPLETED`, `PARTIAL`, `PREPARED` e `INTERRUPTED`, verificando sus hashes y sus estados por candidato/tarea; admitir el directorio no equivale a activar DefTerm sobre todos sus candidatos.

Dos módulos nuevos de producción, sin programas que ejecutar por separado:

- `finterm_defterm.py`: adaptación de las utilidades metodológicas, ejemplos, mensajes, recetas y backend local compartido entre gate/definición.
- `finterm_defterm_store.py`: carga/verificación del contrato 03, preparación, caché, presupuestos, certificado post-gate y consolidación.

`run_defterm(...)` es una función interna de composición para la futura librería única. Sus presupuestos predeterminados son **cero**, y el modelo solo se carga ante un miss que pueda consumir presupuesto. La preselección predeterminada es `preselection_policy="conservative"`. La caché DefTerm debe estar separada de la caché de traducciones y de los inputs/outputs. No se cambia el núcleo 03 ni se agregan dependencias Python nuevas.

Se mantienen las fuentes históricas en el mismo directorio de código y los dos recursos originales:

- `05_sufficiency_gate_fewshot_examples.csv`: diez ejemplos del gate, seis positivos y cuatro negativos.
- `05_fewshot_ejemplos_aprobados.csv`: seis definiciones aprobadas.

`gate_examples=` y `definition_examples=` permiten indicar otras ubicaciones de **esos mismos recursos aprobados**, sin depender de la ubicación del corpus. Se comprueban sus esquemas, aprobación, pertenencia/orden y hashes de los prefijos de mensajes aprobados. Los hashes/rutas de sus anotaciones originales se conservan como procedencia del recurso; no se abren los inventarios, pools o evaluaciones a los que apuntan. No se necesitan los 150 candidatos ni los resultados/manifiestos antiguos.

## Reutilización exacta y adaptación

| Fuente histórica | Reutilización |
|---|---|
| `04_preparar_defterm_contextual_v2.py` | `prepare_inventory` con cantidad dinámica, parsers, perfiles y `build_evidence_text`. |
| `05_defterm_qwen14_sufficiency_gate_fewshot.py` | Prompts, ejemplos proyectados, parser exacto, configuración, `generate_once`, `process_candidate`, determinismo y metadatos de runtime. |
| `05_defterm_qwen14_sufficiency_gate_fewshot_full_v2.py` | `effective_context` y su exclusión de la propia demostración: diez o nueve ejemplos. |
| `05_preparar_defterm_post_gate.py` | Contrato de partición/certificación suficiente separado de la generación, adaptado a certificados por tarea y resultados parciales; su runner fijo no se ejecuta. |
| `05_defterm_qwen14_few_shot_contextual_v2_full.py` | Prompt, seis demostraciones, proyección de mensajes, guard de suficiencia, `process_candidate`, normalización y auditoría de circularidad. |
| `05_consolidar_sentidos2_final.py` | Funciones originales de identidad de representación y linaje, con hashes/rutas de la ejecución nueva. |

Los wrappers completos contienen imports que validan experimentos fijos. Para sus funciones puras se compilan únicamente los nodos AST originales nombrados, **sin reescribir sus cuerpos** ni ejecutar inicializadores/runners experimentales. Los motores de gate y definición tienen globals privados separados; configurar los 150 tokens de definición no cambia los 16 del gate. La evidencia y los few-shot no se copian en otro catálogo mantenido a mano.

Receta conservada: `Qwen/Qwen3-14B`, revisión `40c069824f4251a91eefaf281ebe4c544efd3e18`, NF4/bfloat16 y doble cuantización, seed **42**, determinismo y TF32 desactivado, sin sampling/thinking, un beam y `repetition_penalty=1.1`. Gate: **16 tokens**; definición: **150 tokens**. Una llamada por tarea; ningún retry por error, abstención o circularidad.

### Aislamiento y leave-self-out

Solo se renderizan **etiqueta, alias y etiquetas de P31/P279/P1343**. El término del usuario, Sentido 1, contexto, consultas, descripciones originales y traducciones no son evidencia para el gate ni para definir. Permanecen en la procedencia y se usan para determinar si el enriquecimiento hace falta, pero no se incorporan al prompt.

El gate proyecta únicamente evidencia/decisión de cada ejemplo. La generación utiliza únicamente evidencia/definición de los seis ejemplos aprobados; no recibe los ejemplos negativos del gate.

La identidad de la propia demostración se adapta al texto modelo-visible equivalente, para conservar leave-self-out aunque cambie el término/documento de origen. Se invoca el selector histórico sobre esa identidad, sin modificar los ejemplos: nueve ejemplos si se excluye el propio caso, diez en los demás casos. Se registran ID excluido, IDs retenidos y hash del prefijo. Los seis ejemplos de definición se conservan completos, según el generador histórico.

## Selección de candidatos: no confundir pendiente con ausencia

Se conserva el umbral léxico histórico de más de una palabra como criterio de disponibilidad del fallback, usando el contador ya versionado en 03. **No es una evaluación semántica**. Una palabra se conserva íntegra en 03 y en la consolidación nueva, aunque pueda requerir enriquecimiento.

### Pertinencia nominal previa, sin modelo ni nuevas consultas

Se conservan la forma original del término y las variantes ya registradas por 02 en `search_queries`/`search_strategies`: lema, ortografía y expansiones de siglas justificadas por esa etapa. Se reúnen por **término original exacto**, utilizando todos los candidatos de la entrada 03, antes de aplicar el filtro de IDs del piloto. Esto permite recuperar el lema de la procedencia de otro candidato de la misma consulta y reconocer un candidato directamente identificado aunque no esté incluido entre los IDs que se quieren enriquecer.

No se vuelve a ejecutar spaCy ni se inventa un singular. Si el lema no está registrado en la procedencia disponible, se conserva `lemma_recorded=false`. Las extensiones de contexto y expresiones Cirrus no se consideran variantes del término. La normalización solo aplica NFKC, casefold, espacios y equivalencias de guiones/apóstrofos Unicode: **no** elimina calificadores, fechas, paréntesis, palabras o modificadores, ni utiliza coincidencia parcial de tokens.

Se compara cada variante con el label y cada alias completos. Una coincidencia nominal justifica su consideración preliminar para enriquecer, pero no prueba identidad semántica con el contexto. No se presupone que los otros resultados de búsqueda sean equivalentes.

Dos modos versionados:

- **`conservative`** (`DIRECT_NAMES_AND_QUERY_TEXT_COVERAGE_V1`, predeterminado): si algún candidato de esa consulta coincide directamente por label/alias y ya tiene texto utilizable, se aplaza el enriquecimiento de los demás candidatos que necesiten fallback, incluso si alguno comparte un alias. Se registra `DEFERRED_QUERY_HAS_DIRECT_TEXT`, con todos los IDs/QIDs que aportan ese texto. No se selecciona un QID ganador.
- **`direct_only`** (`DIRECT_NAMES_WITHOUT_QUERY_SUPPRESSION_V1`): desactiva exclusivamente el aplazamiento por cobertura de la consulta. Sigue exigiendo coincidencia nominal directa, evidencia estructurada adecuada y traducciones resueltas. Permite examinar un sentido alternativo directamente identificado sin borrar ningún candidato ni reescribir las cachés.

En ambos modos, un candidato sin coincidencia nominal justificada queda `RELEVANCE_NOT_ESTABLISHED` cuando los demás requisitos del fallback se cumplen. Aparecer en una búsqueda o carecer de descripción nunca basta. Ninguno de esos estados es una decisión del gate, una declaración de insuficiencia semántica ni un descarte irreversible.

Para cada candidato:

1. Si ya hay una representación inglesa disponible de más de una palabra, queda `TEXTUAL_REPRESENTATION_AVAILABLE`; no se genera DefTerm.
2. Con traducciones `PENDING_BUDGET`, `BLOCKED_ADAPTER` o `INFLIGHT_OR_INTERRUPTED`, queda `DEFERRED_TRANSLATION_PENDING`.
3. Con traducciones fallidas, inválidas o incompletas, queda `DEFERRED_TRANSLATION_ERROR`. No se interpreta un fallo técnico como ausencia definitiva.
4. Una fuente no traducida por política también impide declarar ausencia cuando falta una representación prioritaria: `DEFERRED_TRANSLATION_NOT_SELECTED`.
5. Con recuperación Wikidata incompleta relevante para el candidato: `DEFERRED_WIKIDATA_INCOMPLETE`.
6. Sin alias/P31/P279 se conserva la abstención estructural histórica (`NO_CORE_STRUCTURED_EVIDENCE`); una etiqueta o P1343 exploratorio por sí solos no activan el gate.
7. El resto se prepara con la función histórica. Evidencia vacía/no renderizable o QIDs que se filtran al texto se mantienen como `STRUCTURED_EVIDENCE_NOT_RENDERABLE`, sin inventar información.
8. Antes de construir una tarea del gate se aplica la preselección anterior: aplazamiento conservador cuando ya existe texto de un candidato directamente identificado y, en los restantes casos, exigencia de coincidencia nominal directa.

La auditoría `preselection` se conserva tanto en `05_defterm_candidates.jsonl` dentro de `defterm`, como en `05_defterm_partition.jsonl`: política/versión, término original, variantes crudas y normalizadas con procedencia, label/alias coincidentes, estado del lema, estrategias ignoradas y candidatos con texto que justifican el aplazamiento. El estado previo de traducción o evidencia puede ser el motivo principal de no ejecutar; la auditoría nominal permanece disponible igualmente.

El gate admite **exactamente** `SUFFICIENT_EVIDENCE` o `INSUFFICIENT_EVIDENCE`: espacios adicionales, minúsculas, explicaciones o `YES` son errores, no se reparan. Una salida positiva solo autoriza la generación cuando está confirmada en caché y se observó su finalización EOS. Un límite de tokens/parada desconocida se registra como incompletitud técnica; el resultado bruto y el estado del procesador original se conservan.

La partición distingue `SUFFICIENT`, `INSUFFICIENT` y `UNCLASSIFIED` para pendientes/errores/no seleccionados. No se fuerza la clasificación de todo el lote para permitir avanzar a otros candidatos ya certificados. La definición verifica de nuevo el resultado suficiente y su hash inmediatamente antes de reservar y ejecutar su llamada; una columna editada a mano no sustituye ese resultado.

## Caché y estados

Por tarea se guarda un claim exclusivo y un resultado inmutable con hash, usando el escritor atómico/durable histórico. La identidad comprende etapa, QID, evidencia estructurada, mensajes efectivos, ejemplos y receta. La definición añade el certificado del gate. No depende del término/documento, del presupuesto, de la carpeta del run ni del número de candidatos: puede reutilizar inferencias equivalentes entre ejecuciones.

- Caché propia `finterm-defterm-cache-v1`, externa a 03 y a las traducciones.
- Presupuestos independientes `max_gate_calls` y `max_definition_calls`, enteros no negativos; los hits no consumen presupuesto.
- Los resultados se revalidan con los procesadores históricos sin llamar al modelo. Se comprueban request, hash del resultado, modelo/receta y validación derivada.
- Un claim sin resultado queda `INFLIGHT_OR_INTERRUPTED`; no se repite automáticamente la llamada ambigua.
- Los errores confirmados también se reutilizan como errores, sin retries automáticos ni reclasificación a insuficientes.
- `CONTROLLED_TEST_NO_MODEL` tiene identidad distinta del backend real, para que los dobles offline no se reutilicen como resultados de Qwen.
- Los mensajes arbitrarios de excepciones no se guardan; podrían contener configuración o credenciales.
- Los hashes aportan comprobación de integridad, no firma de autenticidad frente a alguien que reescriba todos los archivos y hashes.

La política de preselección **no forma parte del request del modelo** y no modifica sus claves, prompts, ejemplos o parámetros. Cuando una tarea sigue permitida, conserva su identidad y reutiliza la misma inferencia. Si un candidato se aplaza por la política, tampoco se le adjuntan definiciones secundarias de caché, pero sus registros cacheados permanecen intactos.

Para definir se conserva la normalización histórica de espacios: no se corta a la primera frase, no se corrige contenido ni se aumenta especificidad. La circularidad se registra; no provoca una segunda llamada. La consolidación nueva retiene también salidas fallidas/circulares, identificadas como no disponibles para evaluación posterior. `DEFINITION_CIRCULARITY_REVIEW` e `INCOMPLETE_GENERATION` no se presentan como definiciones verificadas.

La suficiencia del gate es un juicio del modelo sobre **la evidencia estructurada**. `fidelity_status="NOT_VERIFIED"` sigue siendo explícito para la definición generada. Tampoco se acredita identidad contextual con el término del corpus.

## Salidas nuevas y trazabilidad

Todas van a un directorio nuevo/vacío; los resultados originales de 03 y los históricos permanecen intactos:

- `05_defterm_gate.jsonl`: tareas, mensajes, respuestas brutas, procesamiento original y metadatos.
- `05_defterm_partition.jsonl`: estado de elegibilidad, decisión y certificado suficiente por candidato.
- `05_defterm_definitions.jsonl`: tareas condicionadas y resultados de definición, también pendientes/errores.
- `05_defterm_candidates.jsonl`: candidatos originales, copia `source_candidate`, estado DefTerm y todos sus IDs de representación.
- `05_defterm_representations.jsonl`: **todas** las representaciones 03 intactas más las salidas DefTerm identificadas por origen/estado/disponibilidad.
- `05_defterm_lineage.jsonl`: archivo/hash/fila/hash de fila y versión de consolidación, usando la función histórica.
- `05_defterm_associations.jsonl`, `05_defterm_terms.jsonl`, `05_defterm_occurrences.jsonl`: copias lógicas intactas de los contratos de 03.
- `05_defterm_source.json`: procedencia y hashes del input y recursos few-shot.
- `05_defterm_manifest.json`: commit final, presupuestos, hits, estados, componentes y hashes de outputs.

`candidate_id`, `term_id`, `document_id`, `occurrence_uid` y QIDs no se sustituyen. Los IDs nuevos de definiciones reutilizan la función histórica en un namespace/versionado y con los hashes **del run actual**. La caché de generación permanece independiente del lote, aunque los IDs de la vista final incluyan el linaje de ese lote.

Los estados globales son `PREPARED` con presupuesto cero y solo pendientes de presupuesto, `PARTIAL` con diferimientos/errores/incompletitud, `INTERRUPTED` y `COMPLETED` para procesamiento terminado. El estado upstream se conserva por separado. Ninguno equivale a enlace final ni a calidad semántica verificada.

## Primera prueba Jupyter H100 — preparada, no ejecutada en esta entrega

Mantener los dos módulos nuevos junto al código existente y los dos CSV few-shot originales. Usar un kernel limpio y el mismo entorno validado Python 3.11/PyTorch CUDA/Transformers/Accelerate/bitsandbytes. Los pesos/tokenizer de la revisión aprobada deben existir: la carga se fuerza a `local_files_only=True`, sin sustituciones ni descargas. No se necesita copiar claves de scripts antiguos.

### 1. Configuración y preview sin inferencia

```python
import os, sys, json
from pathlib import Path
from datetime import datetime, timezone

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
if os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in (None, ":4096:8"):
    raise ValueError("Configura CUBLAS_WORKSPACE_CONFIG=:4096:8 y reinicia el kernel")
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

CODE = Path("/RUTA_REAL_DEL_CODIGO/terminlink_en")
SENSE2 = Path("/RUTA_REAL_OUTPUT_DINAMICO_03")
WORK = Path("/RUTA_PERSISTENTE_EXISTENTE/piloto_defterm")
HUB_CACHE = None  # O la caché hub existente con el snapshot aprobado.
PRESELECTION = "conservative"  # "direct_only" permite otros sentidos nominalmente identificados.
assert CODE.is_dir() and SENSE2.is_dir() and WORK.parent.is_dir()
assert WORK.resolve() != CODE.resolve() and WORK.resolve() != SENSE2.resolve()
assert SENSE2.resolve() not in WORK.resolve().parents and WORK.resolve() not in SENSE2.resolve().parents
WORK.mkdir(exist_ok=True)
SESSION = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
CACHE = WORK / "defterm_cache"
sys.path.insert(0, str(CODE))
from finterm_defterm import QwenDefTerm
from finterm_defterm_store import load_sense2, prepare_defterm, run_defterm

source = load_sense2(SENSE2)
preview = prepare_defterm(source, preselection_policy=PRESELECTION)
print(preview["counts"])
selected = [row["candidate_id"] for row in preview["targets"] if row["eligibility"] == "ELIGIBLE"][:2]
prepared = run_defterm(source, WORK / f"{SESSION}_prepare", CACHE, candidate_ids=selected or None,
                       preselection_policy=PRESELECTION)
print(prepared["status"], prepared["counts"])
assert prepared["execution"]["gate"]["new_attempts"] == 0
```

La selección es una muestra acotada de candidatos ya existentes, no un ranking ni un enlace propuesto. Preferir la salida dinámica del piloto de cinco términos si contiene candidatos elegibles. Cero tareas es un resultado correcto: puede existir ya texto de candidatos directamente identificados, no estar justificada la pertinencia de los secundarios o seguir pendientes las traducciones. Revisar `preselection` y no borrar evidencia para forzar DefTerm.

### 2. Gate real solamente: máximo dos intentos nuevos

```python
if not selected:
    raise ValueError("La preselección no requiere DefTerm: no cargar el modelo ni forzar inferencias")
backend = QwenDefTerm(cache_dir=HUB_CACHE)
gate_run = run_defterm(source, WORK / f"{SESSION}_gate", CACHE,
    candidate_ids=selected, max_gate_calls=2, max_definition_calls=0, backend=backend,
    preselection_policy=PRESELECTION)
print(gate_run["status"], gate_run["execution"], gate_run["counts"])
assert gate_run["execution"]["gate"]["new_attempts"] <= 2
assert gate_run["execution"]["definition"]["new_attempts"] == 0
```

Revisar `05_defterm_gate.jsonl` y `05_defterm_partition.jsonl`: decisiones exactas, nueve/diez ejemplos, EOS, metadatos y errores. Si falta el modelo, `adapter_error` queda explícito y no se descarga nada. Puede ejecutarse `backend.prepare()` por separado en H100 para comprobar sus recursos; carga el modelo, sin generar texto.

### 3. Definir solo positivos cacheados: máximo un intento nuevo

```python
definition_run = run_defterm(source, WORK / f"{SESSION}_definition", CACHE,
    candidate_ids=selected, max_gate_calls=0, max_definition_calls=1, backend=backend,
    preselection_policy=PRESELECTION)
print(definition_run["status"], definition_run["execution"], definition_run["counts"])
assert definition_run["execution"]["gate"]["new_attempts"] == 0
assert definition_run["execution"]["definition"]["new_attempts"] <= 1
source.verify_files()
```

Si los dos casos son insuficientes o el gate tiene errores, **cero definiciones** es el resultado correcto. Para inspeccionar/repetir sin inferencias, usar otro output, la misma caché y ambos presupuestos cero. No editar decisiones o certificados para forzar una definición. Revisar manualmente fidelidad, especificidad y circularidad de las salidas antes de valorar su calidad.

## Verificación offline

```text
python -B tests/run_defterm_offline.py
python -B tests/run_stage03_offline.py
```

El primer runner bloquea red e imports ML reales. Los ejemplos de entrenamiento son los originales; las respuestas de inferencia son dobles explícitos, no resultados reales ni una evaluación semántica. La aceptación real de la carga/generación H100 queda para las celdas anteriores.

Comprobaciones locales de la adaptación inicial, anteriores al filtro de pertinencia:

- **46 pruebas DefTerm offline**, incluyendo ejecución de las tres celdas documentadas con backend controlado y caché de pruebas diferenciada.
- **47 pruebas de regresión de Sentido 2** superadas, sin modificar sus componentes.
- Preparación de tres candidatos reales ya recuperados: tres tareas del gate, cero tareas de definición y **cero inferencias**.
- Comprobación adicional sobre una salida dinámica real con traducciones pendientes: dos candidatos, uno ya representado y otro aplazado; **cero tareas del gate** y las veinte representaciones de origen conservadas.
- Las comprobaciones con datos reales usan inventarios almacenados, no una nueva recuperación ni el piloto reciente de cinco términos. La evidencia temporal de esas preparaciones está en `<LOCAL_WORKSPACE>`.

### Regresión de pertinencia sobre el piloto H100 real

Se usa el archivo privado `piloto_h100_defterm.zip` suministrado por el usuario, con los outputs completos de 03 y la preparación anterior de DefTerm. Se reproduce el inventario de **280 candidatos** y se comprueba que el archivo anterior programaba exactamente estos siete casos:

| Consulta | QID secundario | Label registrado | Candidato directamente identificado con texto |
|---|---|---|---|
| `interest rates` | Q3936435 | rate risk | Q179179 |
| `interest rates` | Q116867879 | theory of interest | Q179179 |
| `Commercial banks` | Q100343504 | Commercial Banks Global Loan III project | Q848507 |
| `Commercial banks` | Q96107414 | commercial banking industry | Q848507 |
| `Commercial banks` | Q130264145 | vacío | Q848507 |
| `Commercial banks` | Q113548978 | commercial banking | Q848507 |
| `European Central Bank` | Q76161279 | European Central Bank 2014 data breach | Q8901 |

Con la política conservadora no se programa ningún gate para esos siete, ni siquiera al solicitar solo sus IDs: los candidatos con texto se detectan en el inventario completo. Se conservan los 280 candidatos, las **3.508 representaciones** de origen, las asociaciones y el ZIP sin cambios. Los controles sin descripción y con nombre directamente justificado siguen alcanzando gate y generación con dobles offline; no son evaluaciones de calidad semántica.

En `Commercial banks` hay varias coincidencias nominales con texto, incluido Q848507. La auditoría registra todas, sin tratarlas como entidades equivalentes ni elegir un ganador; esa ambigüedad corresponde a la comparación posterior. Los siete aplazamientos no se interpretan como decisiones `INSUFFICIENT_EVIDENCE`.

Resultado verificado del replay actual:

- Selección anterior: **7 tareas del gate**; nueva selección: **0 tareas del gate y 0 de definición**.
- Estados nuevos: 221 con texto disponible, 14 sin evidencia estructurada principal, 38 con traducciones pendientes y los siete aplazados por cobertura nominal de su consulta. `PARTIAL` conserva esos pendientes de 03; no significa que se haya ejecutado el gate.
- **64 pruebas DefTerm superadas**, incluidas las cinco pruebas sobre el ZIP real, sin omisiones locales. Se comprobó también cero llamadas/inicializaciones del backend con presupuestos positivos de siete llamadas por etapa.
- **47 pruebas de Sentido 2** superadas. Se verifican identidad de tareas del modelo entre políticas, reutilización cuando procede e integridad byte a byte de las cachés de alternativas aplazadas.
- La preparación nueva, con cero inferencias y sin imports ML, está en `<LOCAL_WORKSPACE>`. Los outputs originales del ZIP no se sustituyen.

El runner incorpora `tests/test_defterm_preselection.py`. Para el replay privado se utiliza el ZIP de la raíz o la ruta indicada en `FINTERM_DEFTERM_PILOT_ZIP`; si no está disponible, las cinco pruebas privadas se omiten explícitamente, sin sustituirlas por datos inventados. Las demás pruebas controladas se ejecutan sin ese archivo. No se ejecuta código del ZIP ni se consulta Wikidata.
