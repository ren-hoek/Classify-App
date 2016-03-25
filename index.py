from flask import Flask, render_template, request
from forms import ContactForm
import pickle
from classify import clean_occ, bag_of_words, occ_lk_dict

saved_classifier = open('classifier.pickle')
classifier = pickle.load(saved_classifier)
saved_classifier.close()

soc_dict = occ_lk_dict()

app = Flask(__name__)

#Needs setting as CSRF_ENABLED=True for flask-wtf. Keep secret_key, secret
app.secret_key = ''

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/classify', methods=['GET', 'POST'])
def classify():
    form = ContactForm()

    if request.method == 'POST':
        clean_job_title = clean_occ(form.job_title.data)
        features = bag_of_words(clean_job_title)
        soc_code = classifier.classify(features)
        soc_label = soc_dict[soc_code]
        no_occ = int(form.noocc.data)
        a = classifier.prob_classify(features)
        c = [(b, '%.0f - %s' % (b, soc_dict[b]), a.prob(b)) for b in a.samples()]
        list_len = min(no_occ, len(a.samples()))
        form.occdrop.choices = (
        [(e[0], e[1])
            for e in sorted(c, key=lambda pb: pb[2], reverse=True)][:list_len]
        )
        return render_template('classify.html', form=form)
    elif request.method == 'GET':
        return render_template('classify.html', form=form)

if __name__ == '__main__':
    app.run(debug=True)
