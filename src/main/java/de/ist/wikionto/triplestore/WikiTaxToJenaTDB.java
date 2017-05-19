/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package de.ist.wikionto.triplestore;

import java.util.HashMap;
import java.util.Map;
import java.util.Set;

import com.hp.hpl.jena.query.Dataset;
import com.hp.hpl.jena.query.ReadWrite;
import com.hp.hpl.jena.rdf.model.Model;
import com.hp.hpl.jena.rdf.model.Resource;
import com.hp.hpl.jena.tdb.TDBFactory;

import de.ist.wikionto.research.MyLogger;
import de.ist.wikionto.webwiki.model.Classifier;
import de.ist.wikionto.webwiki.model.Element;
import de.ist.wikionto.webwiki.model.Instance;

/**
 *
 * @author Marcel
 */
public class WikiTaxToJenaTDB {

	static MyLogger l = new MyLogger("logs/", "ToJena");

	private static final String URI = "http://myWikiTax.de/";
	private static final String cURI = URI + "Classifier#";
	private static final String iURI = URI + "Instance#";
	private static final String nameURI = URI + "name";
	private static final String depthURI = URI + "depth";
	private static final String ciURI = URI + "classifies";
	private static final String ccURI = URI + "hasSubclassifier";
	private static final String defURI = URI + "definedBy";

	// propose a link uri
	private static final String linkURI = URI + "linksTo";

	// propose a has text uri;
	private static final String textURI = URI + "hasText";

	private static Model model;

	private static Map<String, Resource> classResMap;
	private static Map<String, Resource> instanceResMap;
	private static int maxDepth;

	public static void createTripleStore(Classifier root, int max) {
		maxDepth = max;
		String directory = "./" + root.getName().replaceAll(" ", "");
		Dataset dataset = TDBFactory.createDataset(directory);

		dataset.begin(ReadWrite.WRITE);
		model = dataset.getDefaultModel();
		classResMap = new HashMap<>();
		instanceResMap = new HashMap<>();

		Resource rootResource = model.getResource(cURI + classResMap.size());
		classResMap.put(root.getURIName(), rootResource);
		rootResource.addProperty(model.getProperty(nameURI), root.getName());
		rootResource.addProperty(model.getProperty(depthURI), Integer.toString(root.getMinDepth()));

		transformClassifier(root);
		System.out.println("Remaining after depth filter: #C:" + classResMap.size() + ", #I:" + instanceResMap.size());

		// put outputstream instead of null
		// l.logLn("classifers :" + classResMap.keySet().toString());
		// l.logLn("instances :" + instanceResMap.keySet().toString());

		dataset.commit();
		dataset.end();
	}

	private static void transformClassifier(Classifier classifier) {
		Resource classifierResource = classResMap.get(classifier.getURIName());
		if (!(classifier.getMinDepth() < 6)) {
			System.out.println(classifier.getName());
			return;
		}
		if (null != classifier.getDescription()) {
			Resource descriptionResource;
			l.logLn(classifier.getDescription().getName());
			if (!instanceResMap.containsKey(classifier.getDescription().getURIName())) {
				descriptionResource = model.createResource(iURI + instanceResMap.size());
				descriptionResource.addProperty(model.getProperty(nameURI), classifier.getDescription().getName());
				instanceResMap.put(classifier.getDescription().getURIName(), descriptionResource);
				classifierResource.addProperty(model.getProperty(defURI), descriptionResource);
				transformLinks(classifier.getDescription());
				descriptionResource.addProperty(model.getProperty(textURI), classifier.getDescription().getText());
				transformInstance(classifier.getDescription());
			} else {
				descriptionResource = instanceResMap.get(classifier.getDescription().getURIName());
				classifierResource.addProperty(model.getProperty(defURI), descriptionResource);
			}

		}

		for (Instance instance : classifier.getInstances()) {
			Resource instanceResource;
			if (!instanceResMap.containsKey(instance.getURIName())) {
				// l.logLn(instance.getName());
				instanceResource = model.createResource(iURI + instanceResMap.size());
				instanceResMap.put(instance.getURIName(), instanceResource);
				instanceResource.addProperty(model.getProperty(nameURI), instance.getName());
				classifierResource.addProperty(model.getProperty(ciURI), instanceResource);
				instanceResource.addProperty(model.getProperty(textURI), instance.getText());
				// l.logLn("instance hasText " + instance.getName() + " with
				// length" + instance.getText().length());
				transformLinks(instance);
				transformInstance(instance);
			} else {
				instanceResource = instanceResMap.get(instance.getURIName());
				classifierResource.addProperty(model.getProperty(ciURI), instanceResource);
			}

		}
		classifierResource.addProperty(model.getProperty(textURI), classifier.getText());
		// l.logLn("classifier hasText " + classifier.getName() + " with length"
		// + classifier.getText().length());
		transformLinks(classifier);
		transformSubclassifiers(classifier);
		transformClassifiers(classifier, false);
	}

	private static void transformLinks(Classifier classifier) {
		Resource classifierResource = classResMap.get(classifier.getURIName());
		Set<String> links = classifier.getMainLinks();
		// l.logLn("transform Links : " + classifier.getName() + "
		// (classifier)");
		// System.out.println(classifier.getName() + " : " + links.toString());
		for (String link : links) {
			if (link.contains("Category:")) {
				transformLinkClassifier(classifierResource, link);
			} else {
				transformLinkInstance(classifierResource, link);
			}

		}
	}

	private static void transformLinks(Instance instance) {
		Resource instanceResource = instanceResMap.get(instance.getURIName());
		Set<String> links = instance.getLinks();
		// l.logLn("transform Links : " + instance.getName() + " (instance)");
		// System.out.println(instance.getName() + " : " + links.toString());
		for (String link : links) {
			if (link.contains("Category:")) {
				transformLinkClassifier(instanceResource, link);
			} else {
				transformLinkInstance(instanceResource, link);
			}

		}
	}

	private static void transformLinkInstance(Resource resource, String link) {
		Instance linkInstance = new Instance();
		linkInstance.setName(link);
		if (!instanceResMap.containsKey(linkInstance.getURIName())) {
			Resource linkResource = model.createResource(iURI + instanceResMap.size());
			instanceResMap.put(linkInstance.getURIName(), linkResource);
			linkResource.addProperty(model.getProperty(nameURI), linkInstance.getName());
			resource.addProperty(model.getProperty(linkURI), linkResource);
		} else {
			Resource linkResource = instanceResMap.get(linkInstance.getURIName());
			resource.addProperty(model.getProperty(linkURI), linkResource);
		}
	}

	private static void transformLinkClassifier(Resource resource, String link) {
		Classifier linkClassifier = new Classifier();
		linkClassifier.setName(link.replace("Category:", "").trim());
		if (!classResMap.containsKey(linkClassifier.getURIName())) {
			Resource linkResource = model.createResource(cURI + classResMap.size());
			classResMap.put(linkClassifier.getURIName(), linkResource);
			linkResource.addProperty(model.getProperty(nameURI), linkClassifier.getName());
			resource.addProperty(model.getProperty(linkURI), linkResource);

		} else {
			Resource linkResource = classResMap.get(linkClassifier.getURIName());
			resource.addProperty(model.getProperty(linkURI), linkResource);

		}
	}

	private static void transformSubclassifiers(Classifier classifier) {
		// stops at set maximum depth
		if (classifier.getMinDepth() == maxDepth)
			return;
		for (Classifier subclass : classifier.getSubclassifiers()) {
			Resource subclassifierResource;
			if (!classResMap.containsKey(subclass.getURIName())) {
				subclassifierResource = model.createResource(cURI + classResMap.size());
				classResMap.put(subclass.getURIName(), subclassifierResource);
				subclassifierResource.addProperty(model.getProperty(nameURI), subclass.getName());
				transformClassifier(subclass);
			} else {
				subclassifierResource = classResMap.get(subclass.getURIName());
			}
			Resource typeResource = classResMap.get(classifier.getURIName());
			typeResource.addProperty(model.getProperty(ccURI), subclassifierResource);
			if (!subclassifierResource.hasProperty(model.getProperty(depthURI))) {
				subclassifierResource.addProperty(model.getProperty(depthURI),
						Integer.toString(subclass.getMinDepth()));
				transformClassifier(subclass);
			}
		}
	}

	private static void transformClassifiers(Element element, boolean isInstance) {
		for (String classifier : element.getAllClassifiers()) {
			Resource classifierResource;
			if (!classResMap.containsKey(replaceWhitespaceByUnderscore(classifier))) {
				classifierResource = model.createResource(cURI + classResMap.size());
				classResMap.put(replaceWhitespaceByUnderscore(classifier), classifierResource);
				classifierResource.addProperty(model.getProperty(nameURI), removeUnderscore(classifier));
			} else {
				classifierResource = classResMap.get(replaceWhitespaceByUnderscore(classifier));
			}
			if (isInstance) {
				Resource elementResource = instanceResMap.get(element.getURIName());
				classifierResource.addProperty(model.getProperty(ciURI), elementResource);
			} else {
				Resource elementResource = classResMap.get(element.getURIName());
				classifierResource.addProperty(model.getProperty(ccURI), elementResource);
			}
		}
	}

	private static void transformInstance(Instance entity) {
		transformClassifiers(entity, true);

		/**
		 * List<Information> informationList = entity.getInformationList();
		 * 
		 * for (Information information : informationList) { Resource
		 * informationResource = model.createResource(iURI + informationcount);
		 * informationResource.addProperty(model.getProperty(URI + "name"),
		 * Integer.toString(informationcount));
		 * informationResource.addProperty(model.getProperty(URI + "topic"),
		 * information.getName()); informationcount++; Resource entityResource =
		 * instanceResMap.get(entity.getURIName());
		 * entityResource.addProperty(model.getProperty(URI + "hasInformation"),
		 * informationResource);
		 * 
		 * transformInformation(information, informationResource); }
		 **/
	}

	/**
	 * private void transformInformation(Information information, Resource
	 * informationResource) { List<Property> properties =
	 * information.getProperties(); for (Property property : properties) {
	 * Resource propertyResource = model.createResource(pURI + propertycount);
	 * propertycount++; propertyResource.addProperty(model.getProperty(URI +
	 * "name"), filterHTML(property.getName()));
	 * propertyResource.addProperty(model.getProperty(URI + "value"),
	 * filterHTML(property.getValue()));
	 * informationResource.addProperty(model.getProperty(URI + "hasProperty"),
	 * propertyResource); } }
	 * 
	 * private String filterHTML(String text) { String result =
	 * Jsoup.parse(text).text().trim(); return removeLteGte(result); }
	 * 
	 * private String removeLteGte(String text) { return text.replaceAll("<",
	 * "").replaceAll(">", ""); }
	 **/

	private static String removeUnderscore(String supercat) {
		return supercat.replaceAll("_", " ");
	}

	private static String replaceWhitespaceByUnderscore(String supercat) {
		return supercat.replaceAll(" ", "_");
	}

}
