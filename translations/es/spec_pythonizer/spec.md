# Ethereum 2.0 Fase 0 -- La Beacon Chain

**AVISO**: Este documento es un trabajo en progreso para investigadores e implementadores. Refleja cambios recientes en la especificación y tiene prioridad sobre la implementación de prueba de concepto en Python [[python-poc]](#ref-python-poc).

## Introducción

Este documento representa la especificación para la Fase 0 de Ethereum 2.0 -- La Beacon Chain (Cadena de Baliza).

En el núcleo de Ethereum 2.0 hay una cadena del sistema llamada "beacon chain". La beacon chain almacena y gestiona el registro de [validadores](#dfn-validator). En las fases iniciales de implementación de Ethereum 2.0, el único mecanismo para convertirse en [validador](#dfn-validator) es realizar una transacción unidireccional de ETH a un contrato de depósito en Ethereum 1.0. La activación como [validador](#dfn-validator) ocurre cuando los recibos de depósito de Ethereum 1.0 son procesados por la beacon chain, se alcanza el saldo de activación, y después de un proceso de cola. La salida es voluntaria o forzada como penalización por mal comportamiento.

La fuente principal de carga en la beacon chain son las "atestaciones" (attestations). Las atestaciones son votos de disponibilidad para un bloque de fragmento (shard block), y simultáneamente votos de prueba de participación (PoS) para un bloque de la beacon chain. Un número suficiente de atestaciones para el mismo bloque de fragmento crea un "enlace cruzado" (crosslink), confirmando el segmento del fragmento hasta ese bloque en la beacon chain. Los enlaces cruzados también sirven como infraestructura para la comunicación asíncrona entre fragmentos.

## Notación

Los fragmentos de código que aparecen en `este estilo` deben interpretarse como código Python.

## Terminología

* **Validator** <a id="dfn-validator"></a> - un participante registrado en la beacon chain. Puede convertirse en uno enviando Ether al contrato de depósito de Ethereum 1.0.
* **Active validator** <a id="dfn-active-validator"></a> - un participante activo en el consenso de Ethereum 2.0 invitado a, entre otras cosas, proponer y atestiguar bloques y votar por enlaces cruzados.
* **Committee** - un subconjunto muestreado (pseudo-) aleatoriamente de [validadores activos](#dfn-active-validator). Cuando se hace referencia a un comité colectivamente, como en "este comité atestigua X", se asume que significa "algún subconjunto de ese comité que contiene suficientes [validadores](#dfn-validator) para que el protocolo reconozca que representa al comité".
* **Proposer** - el [validador](#dfn-validator) que crea un bloque de la beacon chain.
* **Attester** - un [validador](#dfn-validator) que es parte de un comité que necesita firmar un bloque de la beacon chain mientras crea simultáneamente un enlace (crosslink) a un bloque de fragmento reciente en una cadena de fragmentos particular.
* **Beacon chain** - la cadena PoS central que es la base del sistema de sharding.
* **Shard chain** - una de las cadenas en las que tienen lugar las transacciones de los usuarios y se almacenan los datos de las cuentas.
* **Block root** - una raíz Merkle de 32 bytes de un bloque de la beacon chain o de un bloque de fragmento. Anteriormente llamado "hash de bloque".
* **Crosslink** - un conjunto de firmas de un comité que atestigua un bloque en una cadena de fragmentos, que puede incluirse en la beacon chain. Los enlaces cruzados son el medio principal por el cual la beacon chain "aprende" sobre el estado actualizado de las cadenas de fragmentos.
* **Slot** - un período durante el cual un proponente tiene la capacidad de crear un bloque de la beacon chain y algunos atestiguadores tienen la capacidad de realizar atestaciones.
* **Epoch** - un lapso alineado de slots durante el cual todos los [validadores](#dfn-validator) tienen exactamente una oportunidad de hacer una atestación.
* **Finalized**, **justified** - ver finalización Casper FFG [[casper-ffg]](#ref-casper-ffg).
* **Withdrawal period** - el número de slots entre la salida de un [validador](#dfn-validator) y el momento en que el saldo del [validador](#dfn-validator) puede ser retirado.
* **Genesis time** - el tiempo Unix del bloque génesis de la beacon chain en el slot 0.

*(Traducción parcial - Ver documento original en inglés para el resto de la especificación)*

# Referencias

Esta sección se divide en referencias Normativas e Informativas. Las referencias normativas son aquellas que deben leerse para implementar esta especificación, mientras que las referencias informativas son simplemente eso, información. Un ejemplo de lo primero podrían ser los detalles de un algoritmo de consenso requerido, y un ejemplo de lo segundo podría ser un puntero a una investigación que demuestre por qué un algoritmo de consenso particular podría ser más adecuado para su inclusión en el estándar que otro.

## Normativas

## Informativas
<a id="ref-casper-ffg"></a> _**casper-ffg**_ </br> &nbsp; _Casper the Friendly Finality Gadget_. V. Buterin and V. Griffith. URL: https://arxiv.org/abs/1710.09437

<a id="ref-python-poc"></a> _**python-poc**_ </br> &nbsp; _Python proof-of-concept implementation_. Ethereum Foundation. URL: https://github.com/ethereum/beacon_chain

# Copyright
Derechos de autor y derechos conexos renunciados vía [CC0](https://creativecommons.org/publicdomain/zero/1.0/).
