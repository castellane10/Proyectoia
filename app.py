import json
import requests
import string
from flask import Flask, request, jsonify
from flask_cors import CORS
from sentence_transformers import SentenceTransformer, util



coincidencias_previas = []

app = Flask(__name__)
CORS(app, resources={r"/chatbot": {"origins": "http://localhost:5173"}})
model = SentenceTransformer('distiluse-base-multilingual-cased-v1')

with open("intents.json", encoding="utf-8") as f:
    intents = json.load(f)


@app.route('/chatbot', methods=['POST', 'OPTIONS'])
def chatbot():
    if request.method == 'OPTIONS':
        return jsonify({'status': 'ok'}), 200
    mensaje_original = request.json.get("mensaje", "")
    print("MENSAJE ORIGINAL:", mensaje_original)

    mensaje_usuario = mensaje_original.lower()
    mensaje_usuario = mensaje_usuario.translate(str.maketrans("", "", string.punctuation.replace('"', '').replace('“', '').replace('”', '')))
    embedding_usuario = model.encode(mensaje_usuario, convert_to_tensor=True)

    mejor_intent = None
    mayor_similitud = 0.0

    try:
        usuarios_api = requests.get("https://pg2backend-production.up.railway.app/api/user", headers={
            "User-Agent": "Mozilla/5.0",
            "Accept": "application/json"
        }).json()
    except:
        usuarios_api = []

    if coincidencias_previas:
        mejor_intent = {"tag": "listar_tareas_usuario"}
        mayor_similitud = 1.0
    
    # Paso previo: si hay coincidencias de nombre, forzamos el intent de usuario
    if not mejor_intent and usuarios_api:
        coincidencias = []
        for usuario in usuarios_api:
            if usuario["nombre"].lower() in mensaje_usuario:
                coincidencias.append(usuario)

        if len(coincidencias) == 1:
            mejor_intent = {"tag": "listar_tareas_usuario"}
            mayor_similitud = 1.0
        elif len(coincidencias) > 1:
            # Seguimos con el análisis semántico, pero guardamos coincidencias
            coincidencias_previas.clear()
            coincidencias_previas.extend(coincidencias)

    import re
    if not mejor_intent or mayor_similitud <= 0.45:
        if re.search(r't[ií]tulo\s*["“](.*?)["”]', mensaje_original.lower()) and \
            re.search(r'descripci[oó]n\s*["“](.*?)["”]', mensaje_original.lower()) and \
            re.search(r'estado\s*(?:"|“)?(pendiente|completado|activo)(?:"|”)?', mensaje_original.lower()):
                mejor_intent = {"tag": "crear_tarea"}
                mayor_similitud = 1.0

    if not mejor_intent:
        for intent in intents["intents"]:
            for pattern in intent["patterns"]:
                embedding_pattern = model.encode(pattern, convert_to_tensor=True)
                similitud = util.cos_sim(embedding_usuario, embedding_pattern).item()

                if similitud > mayor_similitud:
                    mayor_similitud = similitud
                    mejor_intent = intent
                    

    if mejor_intent["tag"] == "listar_tareas" and usuarios_api:
        for usuario in usuarios_api:
            nombre = usuario["nombre"].lower()
            if nombre in mensaje_usuario:
                mejor_intent = {"tag": "listar_tareas_usuario"}
                mayor_similitud = 1.0
                break

    
    if mejor_intent and mayor_similitud > 0.45:
        print("INTENT DETECTADO:", mejor_intent["tag"], "| Similitud:", mayor_similitud)
        headers = {
            "User-Agent": "Mozilla/5.0",
            "Accept": "application/json"
        }

        print("INTENT DETECTADO:", mejor_intent["tag"], "| Similitud:", mayor_similitud)

        if mejor_intent["tag"] == "listar_tareas":
            try:
                response = requests.get("https://pg2backend-production.up.railway.app/api/requests", headers=headers)
                if response.status_code == 200:
                    tareas = response.json()
                    if isinstance(tareas, list) and tareas:
                        respuesta = "Estas son las tareas:\n"
                        for tarea in tareas[:5]:
                            respuesta += f"- {tarea['titulo']} ({tarea['estado']})\n"
                    else:
                        respuesta = "No se encontraron tareas en este momento."
                else:
                    respuesta = "Hubo un problema al consultar las tareas."
            except Exception as e:
                respuesta = f"Ocurrió un error: {str(e)}"
            return jsonify({"respuesta": respuesta})

        elif mejor_intent["tag"] == "listar_tareas_por_estado":
            try:
                estado_filtro = None
                for estado in ["completado", "pendiente", "activo", "completo"]:
                    if estado in mensaje_usuario:
                        if estado.startswith("complet"):
                            estado_filtro = "Completado"
                        elif estado.startswith("pend"):
                            estado_filtro = "Pendiente"
                        elif estado.startswith("activ"):
                            estado_filtro = "Activo"
                        break

                if not estado_filtro:
                    return jsonify({"respuesta": "No entendí qué estado estás buscando (pendiente, activo o completado)."})

                response = requests.get("https://pg2backend-production.up.railway.app/api/requests", headers=headers)
                if response.status_code == 200:
                    tareas = response.json()
                    tareas_filtradas = [
                        tarea for tarea in tareas
                        if tarea["estado"].lower() == estado_filtro.lower()
                    ]
                    if tareas_filtradas:
                        respuesta = f"Tareas con estado {estado_filtro}:\n"
                        for tarea in tareas_filtradas:
                            respuesta += f"- {tarea['titulo']} ({tarea['estado']})\n"
                    else:
                        respuesta = f"No hay tareas con estado {estado_filtro}."
                else:
                    respuesta = "Hubo un problema al consultar las tareas."

            except Exception as e:
                respuesta = f"Ocurrió un error: {str(e)}"
            return jsonify({"respuesta": respuesta})

        elif mejor_intent["tag"] == "listar_tareas_usuario":
            try:
                usuarios_api = requests.get("https://pg2backend-production.up.railway.app/api/user", headers=headers).json()

                nombre_detectado = None
                user_id = None

                if coincidencias_previas:
                    for usuario in coincidencias_previas:
                        if usuario["nombre"].lower() in mensaje_usuario:
                            nombre_detectado = usuario["nombre"]
                            user_id = usuario["id"]
                            coincidencias_previas.clear()
                            break

                if not user_id:
                    coincidencias = []

                    # Paso 1: coincidencia exacta por nombre completo
                    for usuario in usuarios_api:
                        if usuario["nombre"].lower() in mensaje_usuario:
                            coincidencias.append(usuario)

                    # Paso 2: coincidencia por palabra, solo si no hubo exactas
                    if len(coincidencias) == 0:
                        palabras = mensaje_usuario.split()
                        for usuario in usuarios_api:
                            nombre_api = usuario["nombre"].lower()
                            if any(palabra in nombre_api.split() for palabra in palabras):
                                coincidencias.append(usuario)

                    # ⚠️ Si encontramos más de una, intentamos ver si solo una es exacta
                    if len(coincidencias) > 1:
                        coincidencias_exactas = [u for u in coincidencias if u["nombre"].lower() in mensaje_usuario]
                        if len(coincidencias_exactas) == 1:
                            usuario = coincidencias_exactas[0]
                            nombre_detectado = usuario["nombre"]
                            user_id = usuario["id"]
                        elif len(coincidencias_exactas) > 1:
                            coincidencias = coincidencias_exactas

                    if not user_id:
                        if len(coincidencias) == 1:
                            usuario = coincidencias[0]
                            nombre_detectado = usuario["nombre"]
                            user_id = usuario["id"]
                        elif len(coincidencias) > 1:
                            coincidencias_previas.clear()
                            coincidencias_previas.extend(coincidencias)
                            nombres = [u["nombre"] for u in coincidencias]
                            respuesta = (
                                "He encontrado varios usuarios con ese nombre. ¿A quién te refieres?\n"
                                + "\n".join(f"- {n}" for n in nombres)
                            )
                            return jsonify({"respuesta": respuesta})
                        else:
                            respuesta = "No reconocí el nombre del usuario. Intenta con el nombre completo."
                            return jsonify({"respuesta": respuesta})

                    elif len(coincidencias) > 1:
                        coincidencias_previas.clear()
                        coincidencias_previas.extend(coincidencias)
                        nombres = [u["nombre"] for u in coincidencias]
                        respuesta = (
                            "He encontrado varios usuarios con ese nombre. ¿A quién te refieres?\n"
                            + "\n".join(f"- {n}" for n in nombres)
                        )
                        return jsonify({"respuesta": respuesta})

                    else:
                        respuesta = "No reconocí el nombre del usuario. Intenta con el nombre completo."
                        return jsonify({"respuesta": respuesta})

                estado_filtro = None
                for estado in ["completado", "pendiente", "activo"]:
                    if estado in mensaje_usuario:
                        estado_filtro = estado.capitalize()
                        break

                if nombre_detectado:
                    asignaciones = requests.get("https://pg2backend-production.up.railway.app/api/assignments", headers=headers).json()
                    tareas_ids = [a["request"] for a in asignaciones if a["usuario"] == user_id]

                    if tareas_ids:
                        respuesta = f"Tareas de {nombre_detectado.title()}"
                        if estado_filtro:
                            respuesta += f" con estado {estado_filtro}"
                        respuesta += ":\n"

                        tareas_agregadas = 0
                        for tarea_id in tareas_ids:
                            tarea = requests.get(f"https://pg2backend-production.up.railway.app/api/requests/{tarea_id}", headers=headers).json()
                            if not estado_filtro or tarea["estado"].lower() == estado_filtro.lower():
                                respuesta += f"- {tarea['titulo']} ({tarea['estado']})\n"
                                tareas_agregadas += 1

                        if tareas_agregadas == 0:
                            respuesta += "No se encontraron tareas con ese estado."
                    else:
                        respuesta = f"No se encontraron tareas para {nombre_detectado.title()}."
                else:
                    respuesta = "No reconocí el nombre del usuario. Intenta con el nombre completo."

            except Exception as e:
                respuesta = f"Ocurrió un error al buscar tareas del usuario: {str(e)}"
            return jsonify({"respuesta": respuesta})

        if mejor_intent["tag"] == "crear_tarea":
            try:
                import re

                titulo_match = re.search(r'titulo\s*["“](.*?)["”]', mensaje_original.lower())
                descripcion_match = re.search(r'descripci[oó]n\s*["“](.*?)["”]', mensaje_original.lower())
                estado_match = re.search(r'estado\s*(?:"|“)?(pendiente|completado|activo)(?:"|”)?', mensaje_original.lower())

                print("MATCH TÍTULO:", titulo_match)
                print("MATCH DESCRIPCIÓN:", descripcion_match)
                print("MATCH ESTADO:", estado_match)

                titulo = titulo_match.group(1).strip() if titulo_match else ""
                descripcion = descripcion_match.group(1).strip() if descripcion_match else ""
                estado = estado_match.group(1).capitalize() if estado_match else ""

                

                if not titulo or not descripcion or not estado:
                    print("❌ Faltan campos:", titulo, descripcion, estado)
                    return jsonify({
                        "respuesta": "Para crear una tarea necesito el título, la descripción y el estado (pendiente, activo o completado). Intenta escribirlos entre comillas."
                    })

                # Buscar nombre del usuario en mensaje
                usuario_asignado = None
                usuarios_api = requests.get("https://pg2backend-production.up.railway.app/api/user", headers=headers).json()
                for usuario in usuarios_api:
                    if usuario["nombre"].lower() in mensaje_usuario:
                        usuario_asignado = usuario
                        break

                payload = {
                    "titulo": titulo,
                    "descripcion": descripcion,
                    "estado": estado
                }

                response = requests.post("https://pg2backend-production.up.railway.app/api/requests", json=payload, headers=headers)

                if response.status_code in [200, 201]:
                    nueva_tarea = response.json()
                    task_id = nueva_tarea.get("id")

                    if usuario_asignado and task_id:
                        asignar = requests.post(f"https://pg2backend-production.up.railway.app/api/requests/{task_id}/assign", json={
                            "usuario": usuario_asignado["id"],
                            "assign_method": "Directamente"
                        }, headers=headers)

                        if asignar.status_code in [200, 201]:
                            return jsonify({
                                "respuesta": f"✅ Tarea creada y asignada a {usuario_asignado['nombre']}: {titulo} ({estado})"
                            })
                        else:
                            return jsonify({
                                "respuesta": f"✅ Tarea creada, pero hubo un error al asignarla. Código: {asignar.status_code}"
                            })
                    else:
                        return jsonify({
                            "respuesta": f"✅ Tarea creada: {titulo} ({estado}). No se especificó un usuario válido para asignar."
                        })
                else:
                    return jsonify({"respuesta": f"Ocurrió un error al crear la tarea. Código: {response.status_code}"})

            except Exception as e:
                return jsonify({"respuesta": f"Error al procesar la creación: {str(e)}"})

            except Exception as e:
                return jsonify({"respuesta": f"Error al procesar la creación: {str(e)}"})

        else:
            return jsonify({"respuesta": mejor_intent["respuesta"]})
    else:
        return jsonify({"respuesta": "Lo siento, no entendí tu pregunta. ¿Puedes reformularla?"})

if __name__ == '__main__':
    app.run(debug=True)
