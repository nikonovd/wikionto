/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package de.ist.wikionto.webwiki.model;

/**
 *
 * @author Marcel
 */
public class NamedElement {
    
    private String name;
    
    public String getName(){
        return name;
    }
    
    public void setName(String pname){
        name = pname.replaceAll(" ", "_");
    }
}
