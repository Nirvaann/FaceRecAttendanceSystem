import base64
import cv2
import os
from flask import Flask, jsonify,session, request, render_template, redirect, url_for
from datetime import date, datetime
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import joblib

# Defining Flask App
app = Flask(__name__)
app.secret_key = 'ananyaisthebest'

nimgs = 10

# Saving Date today in 2 different formats
datetoday = date.today().strftime("%m_%d_%y")
datetoday2 = date.today().strftime("%d-%B-%Y")


# Initializing VideoCapture object to access WebCam
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


# If these directories don't exist, create them
if not os.path.isdir('Attendance'):
    os.makedirs('Attendance')
if not os.path.isdir('static'):
    os.makedirs('static')
if not os.path.isdir('static/faces'):
    os.makedirs('static/faces')
if f'Attendance-{datetoday}.csv' not in os.listdir('Attendance'):
    with open(f'Attendance/Attendance-{datetoday}.csv', 'w') as f:
        f.write('Name,Roll,Time')


# get a number of total registered users
def totalreg():
    return len(os.listdir('static/faces'))


# extract the face from an image
def extract_faces(img):
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_points = face_detector.detectMultiScale(gray, 1.2, 5, minSize=(20, 20))
        return face_points
    except:
        return []


# Identify face using ML model
def identify_face(facearray):
    model_path = 'static/face_recognition_model.pkl'
    if not os.path.exists(model_path):
        raise Exception("No trained model available.")
    model = joblib.load(model_path)
    return model.predict(facearray)


# A function which trains the model on all the faces available in faces folder
def train_model():
    faces = []
    labels = []
    userlist = os.listdir('static/faces')
    if not userlist:
        # If no users, remove model file if it exists and return
        model_path = 'static/face_recognition_model.pkl'
        if os.path.exists(model_path):
            os.remove(model_path)
        return
    for user in userlist:
        for imgname in os.listdir(f'static/faces/{user}'):
            img = cv2.imread(f'static/faces/{user}/{imgname}')
            resized_face = cv2.resize(img, (50, 50))
            faces.append(resized_face.ravel())
            labels.append(user)
    faces = np.array(faces)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(faces, labels)
    joblib.dump(knn, 'static/face_recognition_model.pkl')


# Extract info from today's attendance file in attendance folder
def extract_attendance():
    df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')
    names = df['Name']
    rolls = df['Roll']
    times = df['Time']
    l = len(df)
    return names, rolls, times, l


# Add Attendance of a specific user
def add_attendance(name):
    username = name.split('_')[0]
    userid = name.split('_')[1]
    current_time = datetime.now().strftime("%H:%M:%S")

    df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')
    if int(userid) not in list(df['Roll']):
        with open(f'Attendance/Attendance-{datetoday}.csv', 'a') as f:
            f.write(f'\n{username},{userid},{current_time}')


## A function to get names and roll numbers of all users
def getallusers():
    userlist = os.listdir('static/faces')
    names = []
    rolls = []
    l = len(userlist)

    for i in userlist:
        name, roll = i.split('_')
        names.append(name)
        rolls.append(roll)

    return userlist, names, rolls, l


## A function to delete a user folder 
def deletefolder(duser):
    pics = os.listdir(duser)
    for i in pics:
        os.remove(duser+'/'+i)
    os.rmdir(duser)


################## ROUTING FUNCTIONS #########################

# Our main page
@app.route('/')
def home():
    names, rolls, times, l = extract_attendance()
    return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2)


## List users page
@app.route('/listusers')
def listusers():
    userlist, names, rolls, l = getallusers()
    return render_template('listusers.html', userlist=userlist, names=names, rolls=rolls, l=l, totalreg=totalreg(), datetoday2=datetoday2)


## Delete functionality
@app.route('/deleteuser', methods=['GET'])
def deleteuser():
    duser = request.args.get('user')
    deletefolder('static/faces/' + duser)
    # If all faces are deleted, remove the trained model file
    if len(os.listdir('static/faces/')) == 0:
        if os.path.exists('static/face_recognition_model.pkl'):
            os.remove('static/face_recognition_model.pkl')
    else:
        train_model()
    return redirect(url_for('home'))

# A function to add a new user.
# This function will run when we add a new user.
@app.route('/add', methods=['GET', 'POST'])
def add():
    newusername = request.form['newusername']
    newuserid = request.form['newuserid']
    userimagefolder = 'static/faces/'+newusername+'_'+str(newuserid)
    if not os.path.isdir(userimagefolder):
        os.makedirs(userimagefolder)
    i, j = 0, 0
    cap = cv2.VideoCapture(0)
    while 1:
        _, frame = cap.read()
        faces = extract_faces(frame)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 20), 2)
            cv2.putText(frame, f'Images Captured: {i}/{nimgs}', (30, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 20), 2, cv2.LINE_AA)
            if j % 5 == 0:
                name = newusername+'_'+str(i)+'.jpg'
                cv2.imwrite(userimagefolder+'/'+name, frame[y:y+h, x:x+w])
                i += 1
            j += 1
        if j == nimgs*5:
            break
        cv2.imshow('Adding new User', frame)
        if cv2.waitKey(1) == 27:
            break
    cap.release()
    cv2.destroyAllWindows()
    print('Training Model')
    train_model()
    
    # Redirect to the home page after adding the user
    return redirect(url_for('home'))


# ...existing code...

# Add this near your other cascade initializations
eye_detector = cv2.CascadeClassifier('haarcascade_eye.xml')

@app.route('/start')
def start_attendance():
    cap = cv2.VideoCapture(0)
    recognized = set()
    mess = ""
    import time
    start_time = time.time()
    face_found = False
    TIMEOUT = 5  # seconds

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        faces = extract_faces(frame)
        if len(faces) == 0:
            cv2.putText(frame, "No face detected", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow('Attendance', frame)
            # Check for timeout
            if time.time() - start_time > TIMEOUT:
                mess = "No face detected in camera. Please try again."
                break
            if cv2.waitKey(100) == 27:  # ESC to exit
                break
            continue
        else:
            face_found = True
            start_time = time.time()  # Reset timer if face is found

        for (x, y, w, h) in faces:
            face_img = frame[y:y+h, x:x+w]
            gray_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
            eyes = eye_detector.detectMultiScale(gray_face, 1.1, 4)
            if len(eyes) < 2:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
                cv2.putText(frame, "Keep eyes open", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                continue
            if w < 80 or h < 80:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 255), 2)
                cv2.putText(frame, "Readjust position", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)
                continue
            resized_face = cv2.resize(face_img, (50, 50)).ravel().reshape(1, -1)
            try:
                user = identify_face(resized_face)[0]
                if user not in recognized:
                    add_attendance(user)
                    recognized.add(user)
                    mess = f"Attendance marked for: {', '.join(recognized)}"
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, user, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            except:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                cv2.putText(frame, "Access Denied", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                cv2.imshow('Attendance', frame)
                cv2.waitKey(500)
        cv2.imshow('Attendance', frame)
        if cv2.waitKey(100) == 27 or len(recognized) > 0:  # ESC to exit, 100ms per frame
            break
    cap.release()
    cv2.destroyAllWindows()
    names, rolls, times, l = extract_attendance()
    return render_template('home.html', mess=mess, names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2)

@app.route('/admin_login', methods=['GET', 'POST'])
def admin_login():
    mess = ""
    if request.method == 'POST':
        admin_number = request.form['admin_number']
        # Start webcam and capture face for authentication
        cap = cv2.VideoCapture(0)
        authenticated = False
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            faces = extract_faces(frame)
            for (x, y, w, h) in faces:
                face_img = frame[y:y+h, x:x+w]
                resized_face = cv2.resize(face_img, (50, 50)).ravel().reshape(1, -1)
                try:
                    user = identify_face(resized_face)[0]
                    if user == f"admin_{admin_number}":
                        authenticated = True
                        break
                except:
                    continue
            cv2.imshow('Admin Login', frame)
            if cv2.waitKey(1) == 27 or authenticated:
                break
        cap.release()
        cv2.destroyAllWindows()
        if authenticated:
            session['admin_authenticated'] = True
            return redirect(url_for('admin_controls'))
        else:
            mess = "Authentication failed. Try again."
    return render_template('admin_login.html', mess=mess)

@app.route('/admin_controls')
def admin_controls():
    if not session.get('admin_authenticated'):
        return redirect(url_for('admin_login'))
    userlist, names, rolls, l = getallusers()
    users = zip(userlist, names, rolls)
    return render_template('admin_controls.html', users=users)


@app.route('/remove_user', methods=['POST'])
def remove_user():
    if not session.get('admin_authenticated'):
        return redirect(url_for('admin_login'))
    user = request.form['user']
    deletefolder(f'static/faces/{user}')
    # If all faces are deleted, remove the trained model file
    if len(os.listdir('static/faces/')) == 0:
        if os.path.exists('static/face_recognition_model.pkl'):
            os.remove('static/face_recognition_model.pkl')
    else:
        train_model()
    return redirect(url_for('admin_controls'))

@app.route('/add_user_via_webcam', methods=['POST'])
def add_user_via_webcam():
    data = request.get_json()
    images = data.get('images')
    newusername = data.get('newusername')
    newuserid = data.get('newuserid')
    if not images or len(images) < 10 or not newusername or not newuserid:
        return jsonify(success=False, error="Missing data or not enough images"), 400

    try:
        user_folder = f"static/faces/{newusername}_{newuserid}"
        if not os.path.isdir(user_folder):
            os.makedirs(user_folder)

        saved = 0
        for idx, image_data in enumerate(images):
            img_bytes = base64.b64decode(image_data.split(',')[1])
            np_arr = np.frombuffer(img_bytes, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            faces = extract_faces(frame)
            for (x, y, w, h) in faces:
                if w < 80 or h < 80:
                    continue
                face_img = frame[y:y+h, x:x+w]
                img_path = f"{user_folder}/{newusername}_{saved}.jpg"
                cv2.imwrite(img_path, face_img)
                saved += 1
                break  # Save only one face per image
            if saved >= 10:
                break

        train_model()
        return jsonify(success=True)
    except Exception as e:
        return jsonify(success=False, error=str(e)), 500
    

@app.route('/register_admin', methods=['GET', 'POST'])
def register_admin():
    mess = ""
    if request.method == 'POST':
        if request.is_json:
            data = request.get_json()
            admin_name = data.get('admin_name')
            admin_number = data.get('admin_number')
            images = data.get('images')
            if not admin_name or not admin_number or not images or len(images) < 10:
                return jsonify(success=False, error="Missing data or not enough images"), 400

            admin_folder = f'static/faces/admin_{admin_number}'
            if os.path.isdir(admin_folder):
                return jsonify(success=False, error="Admin already exists!")
            os.makedirs(admin_folder)

            saved = 0
            for idx, image_data in enumerate(images):
                img_bytes = base64.b64decode(image_data.split(',')[1])
                np_arr = np.frombuffer(img_bytes, np.uint8)
                frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                faces = extract_faces(frame)
                for (x, y, w, h) in faces:
                    if w < 80 or h < 80:
                        continue
                    face_img = frame[y:y+h, x:x+w]
                    img_path = f"{admin_folder}/admin_{admin_number}_{saved}.jpg"
                    cv2.imwrite(img_path, face_img)
                    saved += 1
                    break  # Save only one face per image
                if saved >= 10:
                    break

            train_model()
            return jsonify(success=True)
        # ...legacy form POST code...
    return render_template('register_admin.html', mess=mess)


@app.route('/usermanagement')
def usermanagement():
    # Simple admin check (replace with real authentication in production)
    if request.args.get('admin') != '1':
        return "Access denied", 403
    userlist, names, rolls, l = getallusers()
    return render_template('usermanagement.html', userlist=userlist, names=names, rolls=rolls, l=l, totalreg=totalreg(), datetoday2=datetoday2)

@app.route('/scan_attendance', methods=['POST'])
def scan_attendance():
    data = request.get_json()
    if not data or 'image' not in data:
        return jsonify(success=False, error="No image data"), 400

    try:
        # Decode base64 image
        image_data = data['image'].split(',')[1]
        img_bytes = base64.b64decode(image_data)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        # Detect faces
        faces = extract_faces(frame)
        if len(faces) == 0:
            return jsonify(success=False, error="No face detected")

        for (x, y, w, h) in faces:
            face_img = frame[y:y+h, x:x+w]
            if w < 80 or h < 80:
                continue
            resized_face = cv2.resize(face_img, (50, 50)).ravel().reshape(1, -1)
            try:
                user = identify_face(resized_face)[0]
                add_attendance(user)
                username, userid = user.split('_')
                return jsonify(success=True, name=username)
            except Exception as e:
                continue

        return jsonify(success=False, error="Face not recognized")
    except Exception as e:
        return jsonify(success=False, error=str(e)), 500

@app.route('/scan_admin_login', methods=['POST'])
def scan_admin_login():
    data = request.get_json()
    if not data or 'image' not in data or 'admin_number' not in data:
        return jsonify(success=False, error="Missing data"), 400

    try:
        image_data = data['image'].split(',')[1]
        img_bytes = base64.b64decode(image_data)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        faces = extract_faces(frame)
        if len(faces) == 0:
            return jsonify(success=False, error="No face detected")

        admin_id = f"admin_{data['admin_number']}"
        for (x, y, w, h) in faces:
            face_img = frame[y:y+h, x:x+w]
            if w < 80 or h < 80:
                continue
            resized_face = cv2.resize(face_img, (50, 50)).ravel().reshape(1, -1)
            try:
                user = identify_face(resized_face)[0]
                if user == admin_id:
                    name = user.replace('admin_', '')
                    # Set session for admin authentication
                    session['admin_authenticated'] = True
                    # Return redirect URL for admin controls
                    return jsonify(success=True, name=name, redirect_url=url_for('admin_controls'))
            except Exception as e:
                continue

        return jsonify(success=False, error="Face not recognized")
    except Exception as e:
        return jsonify(success=False, error=str(e)), 500

if __name__ == '__main__':
    app.run(debug=True)
