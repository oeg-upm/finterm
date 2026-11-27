# Protocolo concreto de validación full en Jupyter H100

**Estado actual: pendiente de ejecución real.** El notebook distribuido no tiene
outputs ni celdas ejecutadas. Las comprobaciones Windows no validan CUDA ni LLM.

## Preparación

1. Crear/seleccionar un entorno Python 3.11 con `requirements/h100-cu126.txt`.
2. Instalar el paquete y `en_core_web_sm==3.8.0` según INSTALL.md.
3. Reutilizar o aprovisionar por separado los dos modelos revisionados. Verificar
   mediante `finterm-link doctor --require-h100 --hub-cache ... --embedding-resources ...`.
4. Seleccionar el kernel del entorno. El intérprete histórico del equipo es una
   referencia, no una dependencia del paquete.
5. Liberar modelos retenidos por otros kernels/procesos. El padre del enlazador
   no carga modelos; sus tres workers GPU son secuenciales.

Definir en el entorno del kernel rutas reales propias:

```python
import os
os.environ["FINTERM_PROJECT"] = "/CHECKOUT_DEL_ENLAZADOR"
os.environ["FINTERM_ACCEPTANCE_WORK"] = "/ALMACENAMIENTO_PERSISTENTE/aceptaciones"
os.environ["FINTERM_QWEN14_CACHE"] = "/CACHE_HF_EXISTENTE"
os.environ["FINTERM_EMBEDDING_RESOURCES"] = "/RECURSO_EMBEDDING_EXISTENTE"
os.environ["FINTERM_WIKIMEDIA_CONTACT"] = "mailto:research-group@example.org"
```

Sustituir todos los valores de ejemplo. Las variables de autenticación HF,
cuando fueran necesarias para aprovisionamiento, son externas al código/JSON.

## Ejecutar `H100_FULL_VALIDATION.ipynb`

El notebook:

- Comprueba entorno, contratos, few-shot y recursos, sin cargar modelos en el padre.
- Crea un directorio nuevo para evitar confundir hits antiguos con ejecución nueva.
- Conserva `environment.freeze.txt` del entorno real.
- Procesa texto/términos de `examples/financial_cases.json` con `mode="full"`,
  policies historical/conservative y presupuestos explícitos finitos.
- Presenta estados, contadores de inferencias y Top 3/Top 5 por aparición.
- Valida los manifiestos, identidades, asociaciones, pares y MAX originales.
- Exige al menos una traducción contextualizada disponible, un gate suficiente,
  una definición disponible y embeddings puntuados **de producción con intentos
  nuevos**, además de Top 3/5.
- Guarda `h100_full_acceptance.json` y falla expresamente si falta una capacidad.

El informe nunca acepta namespaces controlados ni `archived_replay` como GPU.
Tampoco interpreta Top 1 como un enlace validado.

## Casos adicionales y decisiones metodológicas

Los casos financieros pueden tener descripciones inglesas suficientes y no
requerir DefTerm. Se incluyen consultas naturales específicas, derivadas de
evidencia de los ejemplos aprobados, para ampliar la oportunidad de observar
gate/definición. Solo se suministran textos/términos: **ningún QID candidato se
inyecta como entrada**. Siguen pasando por 01 y recuperación 02 normal.

Algunos probes pueden coincidir con demostraciones few-shot. Son smoke tests de
ejecución, no un conjunto independiente para medir calidad. El gate conserva
su leave-self-out original; no se cambia ningún prompt o demostración.

El estado de Wikidata puede haber cambiado. Si los probes ya tienen texto útil,
la abstención de DefTerm es correcta. Si la evidencia es insuficiente, su gate
debe rechazar. En ambos casos no se habrá demostrado todavía una definición real.
Añadir casos genuinos con evidencia estructurada apropiada a otro JSON con el
mismo formato y continuar el protocolo; no borrar descripciones, alterar relaciones,
forzar una decisión SUFFICIENT ni cambiar la política para lograr el contador.

Si se agota el presupuesto, aumentarlo explícitamente y crear otro run con las
mismas cachés. Los errores ya comprometidos o claims ambiguos no se reintentan
automáticamente. El informe mantiene tareas pendientes/fallidas.

## Qué significa aprobar

`full_runtime_acceptance_passed=true` demuestra que los caminos de producción
se ejecutaron realmente y produjeron las salidas requeridas. Se informa aparte
`all_requested_tasks_completed`, que puede ser falso por presupuestos, errores o
abstenciones pendientes. No se valida con ello fidelidad, precisión de enlaces
o calidad semántica de las definiciones. Los estados PARTIAL no se ocultan.

Guardar y revisar: manifiestos de las fases, `integration_manifest.json`, jobs y
logs GPU, `h100_full_acceptance.json`, freeze y recursos/modelos revisionados.
Los contadores por fase distinguen `new_attempts`/`cache_hits`; los registros
individuales mantienen modelo, receta, mensajes, resultado, EOS y hashes.

## Warm-cache y memoria

Después de demostrar los intentos nuevos, repetir un caso con el mismo `work_dir`:
debe crecer la reutilización y disminuir la inferencia nueva. Comparar los textos
y rankings disponibles, presupuestos y estados; una ejecución parcial anterior
puede completar nuevas tareas y cambiar legítimamente el conjunto de textos MAX.

En el servidor comprobar con las herramientas disponibles que cada worker termina
y libera memoria antes del siguiente. Validar también el bloqueo GPU con otra
ejecución cooperante. La coordinación con extractores externos no forma parte
de esta versión y no se importan sus modelos.
