package com.gwjjj.weiborumor.Business;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.google.gson.Gson;
import com.gwjjj.weiborumor.Utils.RedictUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.ResponseEntity;
import org.springframework.stereotype.Controller;
import org.springframework.util.LinkedMultiValueMap;
import org.springframework.util.MultiValueMap;
import org.springframework.util.StringUtils;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.ResponseBody;
import org.springframework.web.client.RestTemplate;
import org.springframework.web.servlet.view.RedirectView;

import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpSession;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;


@Controller
@RequestMapping("/root")
public class LoginBusiness {
    @Value("${weibo.clientId}")
    private String clientId;
    @Value("${weibo.clientSecret}")
    private String clientSecret; //  应用密码
    @Value("${weibo.redirectUri}")
    private String redirectUri;  //  回调页面
    @Value("${weibo.homeUri}")
    private String homeUri;  //  主页面
    Logger logger = LoggerFactory.getLogger(Logger.class);

    @ResponseBody
    @RequestMapping("/login")
    private RedirectView weiBoCallBack(HttpServletRequest req,String code) {
        logger.info("login =================  login get code : " + code);
        String url="https://api.weibo.com/oauth2/access_token?client_id="+clientId+"&client_secret="+clientSecret+"&grant_type=authorization_code&redirect_uri="+redirectUri+"&code="+code;
        logger.info("login =================  user info url : " + url);
        MultiValueMap<String, String> map= new LinkedMultiValueMap();
        RestTemplate restTemplate = new RestTemplate();
        ResponseEntity<String> stringResponseEntity = restTemplate.postForEntity(url, map, String.class);
        String body = stringResponseEntity.getBody();
        ObjectMapper objectMapper = new ObjectMapper();
        Map map1 = null;
        try {
            map1 = objectMapper.readValue(body, Map.class);
        } catch (IOException e) {
            e.printStackTrace();
        }
        String access_token = (String) map1.get("access_token");
        String uid = (String) map1.get("uid");
        Map oauthResultMap = new HashMap();
        oauthResultMap.put("access_token",access_token);
        oauthResultMap.put("uid",uid);
        logger.info("login =================  oauth2 get access_token : " + access_token);
        logger.info("login =================  oauth2 get uid : " + uid);
        RedirectView redirectView = RedictUtils.createRedirectView(homeUri);
        req.getSession().setAttribute("access_token",access_token);
        req.getSession().setAttribute("uid",uid);
        return redirectView;
    }

    @ResponseBody
    @RequestMapping("/validate")
    private RedirectView validate(HttpServletRequest req) throws IOException {
        String validUrl = "https://api.weibo.com/oauth2/authorize?" +
                "client_id=" + clientId + "&response_type=code&redirect_uri=" + redirectUri;
        RedirectView redirectView = new RedirectView();
        redirectView.setContextRelative(true);
        redirectView.setUrl(validUrl);
        logger.info("validate =================  rediect url success");
        return redirectView;
    }

    @ResponseBody
    @RequestMapping("/getToken")
    private String getToken(HttpServletRequest req) throws IOException {
        HttpSession session = req.getSession();
        String access_token = "";
        if(!StringUtils.isEmpty(access_token = (String)session.getAttribute("access_token"))){
            String uid = (String)session.getAttribute("uid");
            logger.info("getToken ================= have session info");
            Map map = new HashMap<String,String>();
            map.put("access_token",access_token);
            map.put("uid",uid);
            String result = new Gson().toJson(map);
            return result;
        }
        else{
            logger.info("getToken ================= session null");
            return "";
        }
    }

}
