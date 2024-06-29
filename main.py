#Librerías
from tkinter import *
from PIL import Image, ImageTk #Liberería para interfaz gráfica y manejo de imagenes
import imutils #Librería que nos ayuda en el procesamiento de imagenes
import cv2 #Librería que nos ayuda para captura de video y procesamiento de video
import numpy as np #Librería que nos ayuda para manejo de matrices
from ultralytics import YOLO #Librería que nos ayuda para la detección de objetos
import math #Librería que nos ayuda para funciones matemáticas

#Mostrar imagenes
def imagenes(img):
  img = img

  #Detección de imagen como una matriz de numpy
  img = np.array(img, dtype='uint8') # Sentencia para convertir imagen a una matriz de numpy
  img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) #Se convierte de colores RGB a BGR
  img = Image.fromarray(img) #Convertir la matriz de numpy a imagen PIL

  img_ = ImageTk.PhotoImage(image=img) #Conversión de la imagen PIL a formato compatible con Tkinter
  label_imagen.configure(image=img_) #Configurar la imagen en el label de la ventana de Tkinter
  label_imagen.image = img_ #Se matiene la referencia a dicha imagen para evitar que se elimine la imagen




#Función de escaneo
def escaneo():
  global label_imagen #Variable global para la captura de la imagen
  #Interfaz
  label_imagen = Label(pantalla) #Creación de label para mostrar la imagenes en la interfaz
  label_imagen.place(x=75, y=260) #Colocación del label para que aparezcan las imagenes en la interfaz
  #Leer la videocaptura
  if captura is not None:#Verificar si la captura no viene vacía
    ret, frame = captura.read() #Si no viene vacía se leen los frames de la captura por medio del metodo read
    frame_show = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) #conversión de colores de la imagen de BGR a RGB
    if ret == True: #Verificar si se realizo correctamente la lectura de los frames
      
      resultados = modelo(frame, stream=True, verbose=False) #Utilización del modelo para la 
      #detección de los objetos
      for res in resultados:
        #Box
        boxes = res.boxes #Obtención de las cajas delimitadas de los objetos detectados
        for box in boxes:
          #Delimitadores de los boxes
          x1, y1, x2, y2 = box.xyxy[0] #Se establecen las coordenadas de las cajas delimitadoras
          x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) #Se convierten las coordenadas a enteros

          #Solución de error cuando el objeto esta en los límites de la captura
          #Se ajustan las coordenadas de las delimitaciones de las cajas si 
          # estan fuera de los límites de captura
          if x1 < 0: x1 = 0
          if y1 < 0: y1 = 0
          if x2 < 0: x2 = 0
          if y2 < 0: y2 = 0

          #Clase que detecto
          cls = int(box.cls[0]) #Se establece la clase del objeto que se detecto
          #Confidencia
          conf = math.ceil(box.conf[0]) #Se establece el número de confianza 
          #que se obtuvo al detectar el objeto

          if conf > 0.5: #Si el valor de la confianza es mayor a 0.5 se prosigue
            #con su clasificación, si no, se deshecha la lectura
            #Metal
            if cls == 0:
              #Dibujar rectangulo delimitador
              cv2.rectangle(frame_show, (x1, y1), (x2, y2), (255, 255, 0), 2) #se dibuja un rectangulo
              #Alrededor del objeto detectado y se le asigna un color dependiendo de que objeto se
              #detecto en este caso es el amarillo
              #Agregar texto
              text = f'{nombre_clase[cls]} {int(conf)*100}%' #Se crea el texto que se mostrara al rededor
              #del rectangulo creado para y también el porcentaje de confianza con el que se detecto 
              sizetext = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2) #Se le asigna un tamaño
              #al texto que se quiere mostrar además de la tipografía que tomara
              # Obtener dimensiones del texto y la línea base
              dim = sizetext[0] # Tamaño del texto (ancho y alto)
              baseline = sizetext[1] # Línea base del texto
              #Rectangulo
              cv2.rectangle(frame_show, (x1,y1 - dim[1] - baseline), (x1 + dim[0], y1 + baseline), (0,0,0), cv2.FILLED)
              # Dibujar un rectángulo negro para el fondo del texto
              cv2.putText(frame_show, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
              # Dibujar el texto sobre el rectángulo

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
      im = Image.fromarray(frame_show)  # Convercion de una matriz de numpy a imagen de tipo PIL
      img = ImageTk.PhotoImage(image=im) # Converción de la imagen PIL a formato que acepte tkinter
      
      #Mostar en la ventana de Tkinter
      label_video.configure(image=img) #Configurar la imagen en el label de la ventana de Tkinter
      label_video.image = img #Se matiene la referencia a dicha imagen para evitar que se elimine la imagen
      
      label_video.after(10, escaneo) # Repetir función del escaneo cada 10 milisegundos
    else:
      captura.release() # romper con la captura de video si hay un error

def ventana_principal():
  global modelo, nombre_clase, imagen_metal, imagen_vidrio, imagen_plastico, imagen_carton, imagen_medico
  global captura, label_video, pantalla
  #Se declaran de manera global, para poder utilizarlas en otras funciones
  #Ventana principal
  pantalla = Tk()#Se crea la instancia de TK
  pantalla.title("Proyecto de Inteligencia Articial")#Se le asigna un titulo a la ventana
  pantalla.geometry("996x646")#Se asigna la dimensión de la ventana

  #Fondo de la ventana
  imagen_fondo = PhotoImage(file="setUp/Canva_copia.png") #Asignación de la imagen a la variable imagen_fondo
  backgrround = Label(image=imagen_fondo) #Asinganción de la imagen de fondo a la variable background
  backgrround.place(x=0, y=0, relwidth=1, relheight=1)

  #Modelo
  modelo = YOLO("Modelos/best.pt") #Se declara la varibale modelo, a la cual se le asinga un modelo
  #preentrenado y con la clase YOLO se le pasa la ruta para cargar el modelo YOLO a traves de un archivo

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