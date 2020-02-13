<?php
include "config.php";

$email=$_POST['email'];



$sql1 ="SELECT * FROM patients where email_id='$email' ";
$result=$conn->query($sql1);
	if(mysqli_fetch_assoc($result))
	 {

            $password= rand(10000,100000);
            $hashed_password=md5($password);

            $sql =" UPDATE `patients` SET `password` = '$hashed_password' WHERE `patients`.`email_id` = '$email'";
            $conn->query($sql);
            require_once "Mail.php";                                                // Including mail script

            $from = '<apsitmlproject@gmail.com>';                                   //sender's email id
            //taking email id form html form.
            $subject = 'Password Reset Request';   
            $body = " Hi! ".$email." This is your new password ".$password."\n Please use this to login and do not share it with anyone. \n If this is not you then we request you to change the password anyway! \n Thank you! ";                                  

            $headers = array(                                                      //making an header of mail
            'From' => $from,
            'To' => $email,
            'Subject' => $subject
            );

            $smtp = Mail::factory('smtp', array(                                    // Mail::factory() creates mailer instance // smtp is type of backend  
            'host' => 'ssl://smtp.gmail.com',                               //name of host
            'port' => '465',                                                // port number of smtp server
            'auth' => true,                                                 //SMTP Authentication true. because we need to authenticate with gmail username and password
            'username' => 'apsitmlproject@gmail.com',                       //mail of sender(Only gmail)
            'password' => '#'                                  //password of account
            ));

            $mail = $smtp->send($email, $headers, $body);                              //   object-> send to send the mail,prameters are (recipient,email header,email body)

            if (PEAR::isError($mail)) {                                             //  $mail returns a value on failure. PHP Exception Handling is used.
            echo('<p>' . $mail->getMessage() . '</p>');
            } else {
            echo('<p>Successfully sent.....</p>');
            }
     }

    else 
    {
        echo('<p>Invalid User id...</p>');  
    }
?>



