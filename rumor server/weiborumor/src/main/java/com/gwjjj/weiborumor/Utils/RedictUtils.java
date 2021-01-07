package com.gwjjj.weiborumor.Utils;

import org.springframework.web.servlet.view.RedirectView;

import java.util.Map;

public class RedictUtils {

    public static RedirectView createRedirectView(String url){
        RedirectView redirectView = new RedirectView();
        redirectView.setUrl(url);
        redirectView.setContextRelative(true);
        return redirectView;
    }
}
