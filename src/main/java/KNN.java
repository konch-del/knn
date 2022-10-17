import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.lazy.IBk;
import weka.core.Instance;
import weka.core.Instances;

public class KNN {
    public static BufferedReader readDataFile(String filename) {
        BufferedReader inputReader = null;

        try {
            inputReader = new BufferedReader(new FileReader(filename));
        } catch (FileNotFoundException ex) {
            System.err.println("File not found: " + filename);
        }

        return inputReader;
    }

    public static void main(String[] args) throws Exception {
        //читаем файл
        BufferedReader datafile = readDataFile("dataset.txt");

        Instances data = new Instances(datafile);
        //столбец для классификации
        data.setClassIndex(data.numAttributes() - 1);
        //запись для классификации
        Instance first = data.instance(0);
        data.delete(0);
        //n = 3
        Classifier ibk = new IBk(3);
        //обучение
        ibk.buildClassifier(data);

        System.out.println(ibk);

        Evaluation eval = new Evaluation(data);
        eval.evaluateModel(ibk, data);
        /** Print the algorithm summary */
        System.out.println("** KNN **");
        System.out.println(eval.toSummaryString());
        System.out.println(eval.toClassDetailsString());
        System.out.println(eval.toMatrixString());
        //классификация для модели по первой записи
        System.out.println(ibk.classifyInstance(first));
    }
}