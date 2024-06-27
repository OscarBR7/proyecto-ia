#Librerías
from tkinter import *
from PIL import Image, ImageTk
import imutils
import cv2
import numpy as np
from ultralytics import YOLO
import math

#Mostrar imagenes
def imagenes(img):
  img = img

  #Detección de imagen como una matriz de numpy
  img = np.array(img, dtype='unit8')
  img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
  img = Image.fromarray(img)

  img_ = ImageTk.PhotoImage(image=img)
  label_imagen.configure(image=img_)
  label_imagen.image = img_




#Función de escaneo
def escaneo():
  global label_imagen
  #Interfaz
  label_imagen = Label(pantalla)
  label_imagen.place(x=75, y=260)
  #Leer la videocaptura
  if captura is not None:#Verificar si la captura no viene vacía
    ret, frame = captura.read() #Si no viene vacía se leen los frames de la captura por medio del metodo read
    frame_show = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    if ret == True: #Verificar si se realizo correctamente la lectura de los frames
      
      resultados = modelo(frame, stream=True, verbose=False)
      for res in resultados:
        #Box
        boxes = res.boxes
        for box in boxes:
          #Delimitadores de los boxes
          x1, y1, x2, y2 = box.xyxy[0]
          x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

          #Solución de error cuando el objeto esta en los límites de la captura
          if x1 < 0: x1 = 0
          if y1 < 0: y1 = 0
          if x2 < 0: x2 = 0
          if y2 < 0: y2 = 0

          #Clase que detecto
          cls = int(box.cls[0])
          #Confidencia
          conf = math.ceil(box.conf[0])

          if conf > 0.5:
            #Metal
            if cls == 0:
              #Dibujar rectangulo delimitador
              cv2.rectangle(frame_show, (x1, y1), (x2, y2), (255, 255, 0), 2)
              #Agregar texto
              text = f'{nombre_clase[cls]} {int(conf)*100}%'
              sizetext = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
              dim = sizetext[0]
              baseline = sizetext[1]
              #Rectangulo
              cv2.rectangle(frame_show, (x1,y1 - dim[1] - baseline), (x1 + dim[0], y1 + baseline), (0,0,0), cv2.FILLED)
              cv2.putText(frame_show, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

              #Imagen
              imagenes(imagen_metal)
            #Vidrio
            if cls == 1:
              #Dibujar rectangulo delimitador
              cv2.rectangle(frame_show, (x1, y1), (x2, y2), (255, 255, 255), 2)
              #Agregar texto
              text = f'{nombre_clase[cls]} {int(conf)*100}%'
              sizetext = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
              dim = sizetext[0]
              baseline = sizetext[1]
              #Rectangulo
              cv2.rectangle(frame_show, (x1,y1 - dim[1] - baseline), (x1 + dim[0], y1 + baseline), (0,0,0), cv2.FILLED)
              cv2.putText(frame_show, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
              
              #Imagen
              imagenes(imagen_vidrio)
            #Plastico
            if cls == 2:
              #Dibujar rectangulo delimitador
              cv2.rectangle(frame_show, (x1, y1), (x2, y2), (0, 0, 255), 2)
              #Agregar texto
              text = f'{nombre_clase[cls]} {int(conf)*100}%'
              sizetext = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
              dim = sizetext[0]
              baseline = sizetext[1]
              #Rectangulo
              cv2.rectangle(frame_show, (x1,y1 - dim[1] - baseline), (x1 + dim[0], y1 + baseline), (0,0,0), cv2.FILLED)
              cv2.putText(frame_show, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
              
              #Imagen
              imagenes(imagen_plastico)
            #Carton
            if cls == 3:
              #Dibujar rectangulo delimitador
              cv2.rectangle(frame_show, (x1, y1), (x2, y2), (150, 150, 150), 2)
              #Agregar texto
              text = f'{nombre_clase[cls]} {int(conf)*100}%'
              sizetext = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
              dim = sizetext[0]
              baseline = sizetext[1]
              #Rectangulo
              cv2.rectangle(frame_show, (x1,y1 - dim[1] - baseline), (x1 + dim[0], y1 + baseline), (0,0,0), cv2.FILLED)
              cv2.putText(frame_show, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (150, 150, 150), 2)
              
              #Imagen
              imagenes(imagen_carton)
            #Medico
            if cls == 4:
              #Dibujar rectangulo delimitador
              cv2.rectangle(frame_show, (x1, y1), (x2, y2), (255, 0, 0), 2)
              #Agregar texto
              text = f'{nombre_clase[cls]} {int(conf)*100}%'
              sizetext = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
              dim = sizetext[0]
              baseline = sizetext[1]
              #Rectangulo
              cv2.rectangle(frame_show, (x1,y1 - dim[1] - baseline), (x1 + dim[0], y1 + baseline), (0,0,0), cv2.FILLED)
              cv2.putText(frame_show, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

              #Imagen
              imagenes(imagen_medico)
              


      #Redimensionar la captura
      frame_show = imutils.resize(frame_show, width=640)

      #Converción de video al formato pi
      im = Image.fromarray(frame_show)
      img = ImageTk.PhotoImage(image=im)
      
      #Mostar en la ventana de Tkinter
      label_video.configure(image=img)
      label_video.image = img
      label_video.after(10, escaneo)
    else:
      captura.release()

def ventana_principal():
  global modelo, nombre_clase, imagen_metal, imagen_vidrio, imagen_plastico, imagen_carton, imagen_medico
  global captura, label_video, pantalla
  #Se declaran de manera global, para poder utilizarlas en otras funciones
  #Ventana principal
  pantalla = Tk()#Se crea la instancia de TK
  pantalla.title("Proyecto de Inteligencia Articial")#Se le asigna un titulo a la ventana
  pantalla.geometry("1280x720")#Se asigna la dimensión de la ventana

  #Fondo de la ventana
  imagen_fondo = PhotoImage(file="setUp/Canva.png") #Asignación de la imagen a la variable imagen_fondo
  backgrround = Label(image=imagen_fondo) #Asinganción de la imagen de fondo a la variable background
  backgrround.place(x=0, y=0, relwidth=1, relheight=1)

  #Modelo
  modelo = YOLO("Modelos/best.pt") #Se declara la varibale modelo, a la cual se le asinga un modelo
  #preentrenado y con la clase YOLO se le pasa la ruta para cargar el modelo

  #Clases
  nombre_clase = ['Metal', 'Vidrio', 'Plastico', 'Carton', 'Medico'] #Se declara la variable nombre_clase
  #Que almacena una lista con los nombres de las clases que se clasifican

  #Lectura de imagenes
  imagen_metal = cv2.imread("setUP/metal.png")#Se declara la varibale imagen_metal y con la ayuda de 
  #opencv con sobrenombre cv2 se lee la imagen metal.png ubicada en el directorio septUP
  imagen_vidrio = cv2.imread("setUP/vidrio.png")#Se declara la varibale imagen_vidrio y con la ayuda de 
  #opencv con sobrenombre cv2 se lee la imagen metal.png ubicada en el directorio septUP
  imagen_plastico = cv2.imread("setUP/plastico.png")#Se declara la varibale imagen_plastico y con la ayuda de 
  #opencv con sobrenombre cv2 se lee la imagen metal.png ubicada en el directorio septUP
  imagen_carton = cv2.imread("setUP/carton.png")#Se declara la varibale imagen_carton y con la ayuda de 
  #opencv con sobrenombre cv2 se lee la imagen metal.png ubicada en el directorio septUP
  imagen_medico = cv2.imread("setUP/medical.png")#Se declara la varibale imagen_medical y con la ayuda de 
  #opencv con sobrenombre cv2 se lee la imagen metal.png ubicada en el directorio septUP

  #Pantalla para el lector de objetos
  label_video = Label(pantalla) #Se crea el label para el recuadro donse se captura la imagen
  label_video.place(x=319, y=118)#Se le asinga el lugar donde se encontrará la imagen

  #Captura de la cámara
  captura = cv2.VideoCapture(0, cv2.CAP_DSHOW) #Se crea la variable captura que por medio de opencv
  #se realiza la captua, se le pasa el parametro 0 para indicar la camara predeterminada y el segundo
  #parametro indica que es para que windows pueda realizar la captura 
  captura.set(3, 640) #Se declaran los tamaños que tendrá la camara al momento de capturar la imagen
  captura.set(4,480) #Se declaran los tamaños que tendrá la camara al momento de capturar la imagen

  #Escaneo de los objetos
  escaneo()






  #Loop para visualizar la ventana
  pantalla.mainloop()

if __name__ == '__main__': #Definición de ejecución de script principal
  #Si __name__ es igual a main se ejecuta el script correctamente si no, existe un fallo
  ventana_principal()#Llamada a la función principal