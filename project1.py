import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from ultralytics import YOLO
import cv2
import numpy as np


# video dans le meme dossier que le code
video_path = "vid2.mp4"

# modele YOLO pre-trained
model = YOLO("yolov8n.pt")

# les objets qu'on veut afficher
objets_detectes = {"person", "car", "motorcycle"}

# variables pour garder les anciennes lignes
ancienne_ligne_gauche = None
ancienne_ligne_droite = None

# valeur pour stabiliser un peu les lignes
smooth_value = 0.92


def lisser_ligne(nouvelle, ancienne):
    # si on ne trouve pas une nouvelle ligne, on garde l'ancienne
    if nouvelle is None:
        return ancienne

    if ancienne is None:
        return nouvelle

    ligne = []
    for i in range(4):
        valeur = int(smooth_value * ancienne[i] + (1 - smooth_value) * nouvelle[i])
        ligne.append(valeur)

    return tuple(ligne)


def detection_lignes(frame):
    global ancienne_ligne_gauche, ancienne_ligne_droite

    hauteur, largeur = frame.shape[:2]

    gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # masque pour les lignes blanches et jaunes
    masque_blanc = cv2.inRange(gris, 200, 255)
    masque_jaune = cv2.inRange(hsv, (15, 80, 80), (35, 255, 255))

    masque = cv2.bitwise_or(masque_blanc, masque_jaune)

    blur = cv2.GaussianBlur(masque, (5, 5), 0)
    edges = cv2.Canny(blur, 60, 160)

    # zone de recherche des lignes
    zone = np.zeros_like(edges)

    points = np.array([[
        (int(0.22 * largeur), hauteur),
        (int(0.44 * largeur), int(0.62 * hauteur)),
        (int(0.56 * largeur), int(0.62 * hauteur)),
        (int(0.78 * largeur), hauteur)
    ]], np.int32)

    cv2.fillPoly(zone, points, 255)
    edges = cv2.bitwise_and(edges, zone)

    lignes = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        threshold=55,
        minLineLength=90,
        maxLineGap=50
    )

    lignes_gauche = []
    lignes_droite = []

    if lignes is not None:
        for ligne in lignes:
            x1, y1, x2, y2 = ligne[0]

            if x2 == x1:
                continue

            pente = (y2 - y1) / (x2 - x1)

            # supprimer les lignes presque horizontales ou trop verticales
            if abs(pente) < 0.55 or abs(pente) > 2.5:
                continue

            poly = np.polyfit([y1, y2], [x1, x2], 1)
            x_bas = int(poly[0] * hauteur + poly[1])

            if pente < 0 and int(0.20 * largeur) < x_bas < int(0.50 * largeur):
                lignes_gauche.append((x1, y1, x2, y2))

            elif pente > 0 and int(0.50 * largeur) < x_bas < int(0.80 * largeur):
                lignes_droite.append((x1, y1, x2, y2))

    def moyenne_ligne(lignes_liste):
        if len(lignes_liste) == 0:
            return None

        xs = []
        ys = []

        for x1, y1, x2, y2 in lignes_liste:
            xs.append(x1)
            xs.append(x2)
            ys.append(y1)
            ys.append(y2)

        poly = np.polyfit(ys, xs, 1)

        y_bas = hauteur
        y_haut = int(0.63 * hauteur)

        x_bas = int(poly[0] * y_bas + poly[1])
        x_haut = int(poly[0] * y_haut + poly[1])

        return x_bas, y_bas, x_haut, y_haut

    nouvelle_gauche = moyenne_ligne(lignes_gauche)
    nouvelle_droite = moyenne_ligne(lignes_droite)

    # eviter que les deux lignes se croisent
    if nouvelle_gauche is not None and nouvelle_droite is not None:
        if nouvelle_gauche[2] >= nouvelle_droite[2]:
            nouvelle_gauche = None
            nouvelle_droite = None

    ancienne_ligne_gauche = lisser_ligne(nouvelle_gauche, ancienne_ligne_gauche)
    ancienne_ligne_droite = lisser_ligne(nouvelle_droite, ancienne_ligne_droite)

    affichage_lignes = np.zeros_like(frame)

    if ancienne_ligne_gauche is not None:
        cv2.line(
            affichage_lignes,
            (ancienne_ligne_gauche[0], ancienne_ligne_gauche[1]),
            (ancienne_ligne_gauche[2], ancienne_ligne_gauche[3]),
            (255, 0, 0),
            5
        )

    if ancienne_ligne_droite is not None:
        cv2.line(
            affichage_lignes,
            (ancienne_ligne_droite[0], ancienne_ligne_droite[1]),
            (ancienne_ligne_droite[2], ancienne_ligne_droite[3]),
            (255, 0, 0),
            5
        )

    resultat = cv2.addWeighted(frame, 1, affichage_lignes, 0.8, 0)
    return resultat


def detection_feux(frame):
    hauteur, largeur = frame.shape[:2]

    # on cherche les feux seulement dans la partie haute de l'image
    y1_roi = 0
    y2_roi = int(0.50 * hauteur)
    x1_roi = int(0.10 * largeur)
    x2_roi = int(0.90 * largeur)

    roi = frame[y1_roi:y2_roi, x1_roi:x2_roi]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # plages HSV pour les couleurs des feux
    rouge1 = cv2.inRange(hsv, (0, 150, 150), (8, 255, 255))
    rouge2 = cv2.inRange(hsv, (172, 150, 150), (180, 255, 255))
    vert = cv2.inRange(hsv, (45, 130, 130), (85, 255, 255))
    jaune = cv2.inRange(hsv, (20, 130, 130), (32, 255, 255))

    liste_masques = [
        ("RED", rouge1 | rouge2, (0, 0, 255)),
        ("GREEN", vert, (0, 255, 0)),
        ("YELLOW", jaune, (0, 255, 255))
    ]

    feux_trouves = []

    for nom_couleur, masque, couleur_dessin in liste_masques:
        kernel = np.ones((3, 3), np.uint8)

        masque = cv2.morphologyEx(masque, cv2.MORPH_OPEN, kernel)
        masque = cv2.morphologyEx(masque, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(
            masque,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        for contour in contours:
            surface = cv2.contourArea(contour)

            if surface < 45 or surface > 450:
                continue

            x, y, w_box, h_box = cv2.boundingRect(contour)

            if w_box < 6 or h_box < 6 or w_box > 32 or h_box > 32:
                continue

            rapport = w_box / float(h_box)

            if rapport < 0.65 or rapport > 1.45:
                continue

            perimetre = cv2.arcLength(contour, True)

            if perimetre == 0:
                continue

            circularite = 4 * np.pi * surface / (perimetre * perimetre)

            if circularite < 0.50:
                continue

            vrai_x = x + x1_roi
            vrai_y = y + y1_roi

            # petit filtre pour eviter le panneau jaune a gauche
            if nom_couleur == "YELLOW":
                centre_x = vrai_x + w_box // 2
                centre_y = vrai_y + h_box // 2

                if centre_x < int(0.42 * largeur) or centre_x > int(0.75 * largeur):
                    continue

                if centre_y < int(0.25 * hauteur):
                    continue

            feux_trouves.append((vrai_x, vrai_y, w_box, h_box, nom_couleur, couleur_dessin))

    return feux_trouves


# ouverture de la video
cap = cv2.VideoCapture(video_path)
# Save output video
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')

out = cv2.VideoWriter(
    "ADAS_Result.mp4",
    fourcc,
    fps,
    (width, height)
)

if not cap.isOpened():
    print("Erreur: la video vid2.mp4 est introuvable")
    exit()


while True:
    ret, frame = cap.read()

    if not ret:
        break

    # detection des lignes de voie
    frame = detection_lignes(frame)

    # detection YOLO
    resultats = model(frame, conf=0.35, verbose=False)

    for resultat in resultats:
        for boite in resultat.boxes:
            classe = int(boite.cls[0])
            nom_objet = model.names[classe]
            confiance = float(boite.conf[0])

            if nom_objet not in objets_detectes:
                continue

            x1, y1, x2, y2 = map(int, boite.xyxy[0])

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"{nom_objet} {confiance:.2f}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

    # detection des feux tricolores
    feux = detection_feux(frame)

    for x, y, w_box, h_box, couleur_feu, couleur_dessin in feux:
        cv2.rectangle(
            frame,
            (x - 5, y - 5),
            (x + w_box + 5, y + h_box + 5),
            couleur_dessin,
            2
        )

        cv2.putText(
            frame,
            f"FEU: {couleur_feu}",
            (x - 10, y - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            couleur_dessin,
            2
        )

    cv2.putText(
        frame,
        "ADAS - Demo",
        (30, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    cv2.imshow("ADAS - Detection", frame)

    # touche q pour quitter
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()